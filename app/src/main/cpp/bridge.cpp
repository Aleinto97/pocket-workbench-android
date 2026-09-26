#include <jni.h>
#include <atomic>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <ctime>
#include <ucontext.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <cstdint>
#include "llama.h"
#include "ggml-backend.h"
#include "whisper.h"

static std::atomic<bool> stop_requested{false};

// ---------------------------------------------------------------------------
// Native crash + phase tracking. The JVM-side Diag object consumes the files
// we write here after a process death, so a user-reported "app jumped away at
// the end of a generation" can be attributed to a precise native phase.
// ---------------------------------------------------------------------------
static const char * kPhaseNames[] = {
    "idle", "load_model", "alloc_context", "chat_template", "tokenize",
    "prefill", "generating", "stats_build", "stats_callback", "cleanup", "done"
};
static const int kPhaseCount = (int) (sizeof(kPhaseNames) / sizeof(kPhaseNames[0]));
static std::atomic<int> g_phase{0};
static std::atomic<long> g_gen_tokens{0};
static char g_crash_file[512] = {0};
static char g_state_file[512] = {0};
static char g_model_hint[128] = {0};

// Fine-grained journal of the last native call inside the pipeline, so a
// SIGSEGV report says WHICH call was running (sample vs decode vs callback...).
static const char * kOpNames[] = {
    "none", "prefill_decode", "sample", "accept", "to_piece",
    "jni_str", "jni_call", "jni_exc", "gen_decode", "state_save"
};
static const int kOpCount = (int) (sizeof(kOpNames) / sizeof(kOpNames[0]));
static std::atomic<int> g_last_op{0};
static std::atomic<int> g_threads{0};
static std::atomic<int> g_nctx{0};
static std::atomic<int> g_prompt_tokens{0};
static char g_backend_hint[24] = {0};
static void * g_stack_lo = nullptr; // generation-thread stack range, for overflow flagging
static size_t g_stack_size = 0;

// forward decls: the maps helper below is defined before these helpers
static void write_all(int fd, const char * s, size_t n);
static void write_str(int fd, const char * s);
static void write_long(int fd, long v);
static void write_hex(int fd, unsigned long v);

// Async-signal-safe maps lookup: prints "<lib path>+0x<offset>" for the
// library containing addr. Streams /proc/self/maps line by line: an Android
// 16 app can produce far more maps text than the old fixed 32 KB buffer,
// which silently truncated the tail (high-address libs like libart.so or
// libllama.so were never found, so pc/lr printed "?" in 0.2.4).
// No stdio, no malloc, no unbounded buffering.
static void write_lib_for_addr(int fd, uintptr_t addr) {
    int fdm = open("/proc/self/maps", O_RDONLY);
    if (fdm < 0) { write_str(fd, "?"); return; }
    char chunk[8192];
    char line[512];
    size_t llen = 0;
    bool found = false;
    unsigned long best_lo = 0;
    char best_name[256] = {0};
    ssize_t k;
    while (!found && (k = read(fdm, chunk, sizeof(chunk))) > 0) {
        for (ssize_t i = 0; i < k; ++i) {
            char c = chunk[i];
            if (c != '\n') {
                if (llen < sizeof(line) - 1) line[llen++] = c;
                continue;
            }
            line[llen] = 0;
            llen = 0;
            if (!line[0]) continue;
            char * dash = strchr(line, '-');
            if (!dash) continue;
            unsigned long lo = strtoul(line, nullptr, 16);
            unsigned long hi = strtoul(dash + 1, nullptr, 16);
            if (addr >= lo && addr < hi) {
                char * slash = strchr(line, '/'); // first '/' starts the path field
                if (slash) {
                    size_t plen = strlen(slash);
                    if (plen > sizeof(best_name) - 1) plen = sizeof(best_name) - 1;
                    memcpy(best_name, slash, plen);
                    best_name[plen] = 0;
                    best_lo = lo;
                    found = true;
                }
            }
        }
    }
    close(fdm);
    if (found) {
        write_str(fd, best_name);
        write_str(fd, "+0x");
        write_hex(fd, addr - best_lo);
    } else {
        write_str(fd, "?");
    }
}

// Async-signal-safe helpers: only open/write/close and manual int formatting.
static void write_all(int fd, const char * s, size_t n) {
    while (n > 0) { ssize_t k = write(fd, s, n); if (k <= 0) return; s += k; n -= (size_t) k; }
}
static void write_str(int fd, const char * s) { write_all(fd, s, strlen(s)); }
static void write_long(int fd, long v) {
    char b[24]; int i = (int) sizeof(b);
    unsigned long u = v < 0 ? (unsigned long) (-(v + 1)) + 1UL : (unsigned long) v;
    if (u == 0) { write_str(fd, "0"); return; }
    while (u > 0 && i > 0) { b[--i] = (char) ('0' + (u % 10)); u /= 10; }
    if (v < 0 && i > 0) b[--i] = '-';
    write_all(fd, b + i, sizeof(b) - (size_t) i);
}
// Hex formatter for addresses (async-signal-safe). The old report wrote
// "0x" followed by write_long DECIMAL digits, so fault_addr=0x16777184 was
// actually decimal 16777184 = 0xffffe0.
static void write_hex(int fd, unsigned long v) {
    char b[16]; int i = (int) sizeof(b);
    if (v == 0) { write_str(fd, "0"); return; }
    while (v > 0 && i > 0) {
        int d = (int) (v & 0xF);
        b[--i] = (char) (d < 10 ? '0' + d : 'a' + (d - 10));
        v >>= 4;
    }
    write_all(fd, b + i, sizeof(b) - (size_t) i);
}

static void native_crash_handler(int sig, siginfo_t * info, void * uctx) {
    if (g_crash_file[0]) {
        int fd = open(g_crash_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            int p = g_phase.load();
            int op = g_last_op.load();
            write_str(fd, "signal="); write_long(fd, sig);
            write_str(fd, "\nphase="); write_str(fd, kPhaseNames[p >= 0 && p < kPhaseCount ? p : 0]);
            write_str(fd, "\nlast_op="); write_str(fd, kOpNames[op >= 0 && op < kOpCount ? op : 0]);
            write_str(fd, "\ngen_tokens="); write_long(fd, g_gen_tokens.load());
            write_str(fd, "\nprompt_tokens="); write_long(fd, g_prompt_tokens.load());
            write_str(fd, "\nbackend="); write_str(fd, g_backend_hint);
            write_str(fd, "\nthreads="); write_long(fd, g_threads.load());
            write_str(fd, "\nctx="); write_long(fd, g_nctx.load());
            uintptr_t fa = (uintptr_t) (info ? info->si_addr : nullptr);
            write_str(fd, "\nfault_addr=0x"); write_hex(fd, fa);
            int stack_hit = 0;
            if (g_stack_lo && g_stack_size) {
                uintptr_t lo = (uintptr_t) g_stack_lo;
                uintptr_t hi = lo + g_stack_size;
                uintptr_t margin = (uintptr_t) (2UL << 20);
                if (fa >= (lo > margin ? lo - margin : 0) && fa < hi) stack_hit = 1;
            }
            write_str(fd, "\nstack_hit="); write_long(fd, stack_hit);
#ifdef __aarch64__
            ucontext_t * uc = (ucontext_t *) uctx;
            if (uc) {
                uintptr_t pc = (uintptr_t) uc->uc_mcontext.pc;
                uintptr_t lr = (uintptr_t) uc->uc_mcontext.regs[30];
                // raw values FIRST (the maps walk below is the fragile part)
                write_str(fd, "\npc=0x"); write_hex(fd, pc);
                write_str(fd, "\nlr=0x"); write_hex(fd, lr);
                write_str(fd, "\nsp=0x"); write_hex(fd, (uintptr_t) uc->uc_mcontext.sp);
                write_str(fd, "\npc_lib="); write_lib_for_addr(fd, pc);
                write_str(fd, "\nlr_lib="); write_lib_for_addr(fd, lr);
                write_str(fd, "\nregs=");
                for (int r = 0; r < 31; ++r) {
                    write_str(fd, "x"); write_long(fd, r); write_str(fd, "=0x");
                    write_hex(fd, (uintptr_t) uc->uc_mcontext.regs[r]);
                    if (r != 30) write_str(fd, " ");
                }
            }
#endif
            write_str(fd, "\nmodel="); write_str(fd, g_model_hint);
            write_str(fd, "\n");
            close(fd);
        }
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

static void install_native_crash_handlers() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = native_crash_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
}

static void save_phase_state() {
    if (!g_state_file[0]) return;
    FILE * f = fopen(g_state_file, "w");
    if (!f) return;
    int p = g_phase.load();
    fprintf(f, "pid=%d\nphase=%s\ndetail=tokens=%ld model=%s\nts=%lld\nbackend=%s\nthreads=%d\n",
            (int) getpid(), kPhaseNames[p >= 0 && p < kPhaseCount ? p : 0],
            g_gen_tokens.load(), g_model_hint, (long long) time(nullptr) * 1000LL,
            g_backend_hint, g_threads.load());
    fclose(f);
}
static void set_phase(int p) { g_phase.store(p); save_phase_state(); }

// The last model stays resident so repeated generations (for example a
// multi-step agent tool loop) skip the load phase. A fresh context is
// allocated per call, so no KV state leaks between requests.
static llama_model * cached_model = nullptr;
static std::string cached_path;
static int cached_gpu_layers = 0;
static double last_load_ms = 0.0;

// Convert standard UTF-8 into JNI "modified UTF-8" (CESU-8, NUL as C0 80,
// 4-byte sequences as surrogate pairs) so NewStringUTF can never abort or
// garble token text, whatever the model emits.
static void append_modified_utf8(std::string & out, const char * s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char) s[i];
        if (c < 0x80) {
            if (c == 0) { out += (char) 0xC0; out += (char) 0x80; }
            else out += (char) c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < n) {
            out += s[i]; out += s[i + 1]; i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < n) {
            out += s[i]; out += s[i + 1]; out += s[i + 2]; i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < n) {
            uint32_t cp = ((uint32_t) (c & 0x07) << 18) | ((uint32_t) (s[i+1] & 0x3F) << 12)
                        | ((uint32_t) (s[i+2] & 0x3F) << 6) | (uint32_t) (s[i+3] & 0x3F);
            i += 4;
            if (cp >= 0x10000 && cp <= 0x10FFFF) {
                uint32_t v = cp - 0x10000;
                uint16_t pair[2] = { (uint16_t) (0xD800 + (v >> 10)), (uint16_t) (0xDC00 + (v & 0x3FF)) };
                for (int q = 0; q < 2; ++q) {
                    out += (char) (0xE0 | (pair[q] >> 12));
                    out += (char) (0x80 | ((pair[q] >> 6) & 0x3F));
                    out += (char) (0x80 | (pair[q] & 0x3F));
                }
            }
        } else {
            out += (char) 0xEF; out += (char) 0xBF; out += (char) 0xBD; // U+FFFD
            i += 1;
        }
    }
}

// Remember the calling thread's stack range so the crash handler can flag
// stack-overflow faults (SIGSEGV just below the stack base).
static void capture_thread_stack() {
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) != 0) return;
    void * base = nullptr; size_t sz = 0;
    if (pthread_attr_getstack(&attr, &base, &sz) == 0 && base && sz > 0) {
        g_stack_lo = base; g_stack_size = sz;
    }
    pthread_attr_destroy(&attr);
}

static std::string from_java(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * raw = env->GetStringUTFChars(value, nullptr);
    std::string copy(raw);
    env->ReleaseStringUTFChars(value, raw);
    return copy;
}
static void fail(JNIEnv * env, const char * reason) {
    jclass type = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(type, reason);
}
static double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

extern "C" JNIEXPORT void JNICALL
Java_com_pocketworkbench_app_NativeEngine_stop(JNIEnv *, jobject) { stop_requested = true; }

extern "C" JNIEXPORT void JNICALL
Java_com_pocketworkbench_app_NativeEngine_generate(
        JNIEnv * env, jobject, jstring path, jobjectArray roles, jobjectArray contents, jobject callback, jstring log_dir, jint threads_j) {
    stop_requested = false;
    g_gen_tokens.store(0);
    capture_thread_stack();
    std::string model_path = from_java(env, path);
    {
        std::string logs = from_java(env, log_dir);
        if (!logs.empty()) {
            snprintf(g_crash_file, sizeof(g_crash_file), "%s/native_crash.txt", logs.c_str());
            snprintf(g_state_file, sizeof(g_state_file), "%s/native_state.txt", logs.c_str());
            static bool handlers_installed = false;
            if (!handlers_installed) { install_native_crash_handlers(); handlers_installed = true; }
        }
    }
    const char * base = strrchr(model_path.c_str(), '/');
    snprintf(g_model_hint, sizeof(g_model_hint), "%s", base ? base + 1 : model_path.c_str());
    set_phase(1); // load_model
    ggml_backend_load_all();
    const bool gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) || ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    int gpu_layers = gpu ? 99 : 0;
    bool loaded_now = false;
    bool gpu_fallback = false;
    llama_model * model = nullptr;
    if (cached_model && cached_path == model_path) {
        model = cached_model;
        gpu_layers = cached_gpu_layers;
    } else {
        if (cached_model) { llama_model_free(cached_model); cached_model = nullptr; cached_path.clear(); }
        auto load_start = std::chrono::steady_clock::now();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = gpu ? 99 : 0;
        model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model && gpu) { mp.n_gpu_layers = 0; gpu_layers = 0; gpu_fallback = true; model = llama_model_load_from_file(model_path.c_str(), mp); }
        if (!model) { set_phase(10); fail(env, "Cannot load GGUF model. Check format and free memory."); return; }
        cached_model = model; cached_path = model_path; cached_gpu_layers = gpu_layers;
        loaded_now = true;
        last_load_ms = ms_since(load_start);
    }
    set_phase(2); // alloc_context
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    // llama.cpp issue #28878: SIGSEGV in the CPU compute path on Android/aarch64
    // with 6+ threads (thread-count-dependent race, not content-dependent).
    // hardware_concurrency-1 = 7 on 8-core devices sits in the reliably-crashing
    // zone; the JVM side defaults to 4 and exposes a user setting (1..8).
    cp.n_threads = std::clamp(threads_j > 0 ? (int) threads_j : 4, 1, 8);
    cp.n_threads_batch = cp.n_threads; // keep prefill on the same (safe) count
    g_threads.store(cp.n_threads);
    g_nctx.store((int) cp.n_ctx);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx && gpu_layers > 0) {
        // Context allocation failed with the GPU build; fall back to CPU.
        if (cached_model) { llama_model_free(cached_model); cached_model = nullptr; cached_path.clear(); }
        auto load_start = std::chrono::steady_clock::now();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(model_path.c_str(), mp);
        gpu_layers = 0; gpu_fallback = true; loaded_now = true;
        last_load_ms = ms_since(load_start);
        if (model) { cached_model = model; cached_path = model_path; cached_gpu_layers = 0; ctx = llama_init_from_model(model, cp); }
    }
    if (!ctx) { set_phase(10); fail(env, "Cannot allocate model context."); return; }

    jclass callback_type = env->GetObjectClass(callback);
    jmethodID on_token = env->GetMethodID(callback_type, "onToken", "(Ljava/lang/String;)V");
    jmethodID on_stats = env->GetMethodID(callback_type, "onStats", "(Ljava/lang/String;)V");
    bool ok = on_token != nullptr;
    std::string backend = gpu_layers > 0 ? "Vulkan GPU" : "CPU";
    if (gpu_fallback) backend = "CPU (GPU load failed, fallback)";
    snprintf(g_backend_hint, sizeof(g_backend_hint), "%s",
             gpu_layers > 0 ? "vulkan" : (gpu_fallback ? "cpu-fb" : "cpu"));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<std::string> r, c;
    std::vector<llama_chat_message> messages;
    jsize size = env->GetArrayLength(roles);
    for (jsize i = 0; i < size && i < env->GetArrayLength(contents); ++i) {
        auto jr = (jstring) env->GetObjectArrayElement(roles, i);
        auto jc = (jstring) env->GetObjectArrayElement(contents, i);
        r.push_back(from_java(env, jr)); c.push_back(from_java(env, jc));
        env->DeleteLocalRef(jr); env->DeleteLocalRef(jc);
    }
    for (size_t i = 0; i < r.size(); ++i) messages.push_back({r[i].c_str(), c[i].c_str()});
    set_phase(3); // chat_template
    const char * tmpl = llama_model_chat_template(model, nullptr);
    int required = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, nullptr, 0);
    if (required <= 0 || required > 4 * 1024 * 1024) {
        llama_free(ctx); fail(env, "Model chat template is unavailable."); return;
    }
    std::string prompt(required + 1, '\0');
    llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, prompt.data(), prompt.size());
    prompt.resize(required);
    set_phase(4); // tokenize
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n >= 0) { llama_free(ctx); fail(env, "Tokenization failed."); return; }
    std::vector<llama_token> tokens(-n);
    n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (n <= 0 || n >= 3500) {
        llama_free(ctx); fail(env, "Conversation is too long for the 4096-token context."); return;
    }
    tokens.resize(n);
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0xC0FFEE));

    set_phase(5); // prefill
    g_prompt_tokens.store(n);
    std::string stop_reason = "max_tokens";
    auto prefill_start = std::chrono::steady_clock::now();
    for (int offset = 0; ok && offset < n && !stop_requested; offset += 512) {
        int count = std::min(512, n - offset);
        llama_batch batch = llama_batch_get_one(tokens.data() + offset, count);
        g_last_op.store(1); // prefill_decode
        if (llama_decode(ctx, batch) != 0) { ok = false; stop_reason = "context_full"; }
    }
    double prefill_ms = ms_since(prefill_start);
    int gen_tokens = 0;
    auto gen_start = std::chrono::steady_clock::now();
    set_phase(6); // generating
    // Token text is buffered natively and flushed to the JVM in ~100 ms
    // batches (or 96 bytes): ~20x fewer JNI crossings than per-token calls.
    // Flushes never split a multi-byte sequence, and the text is converted
    // to modified UTF-8 before NewStringUTF, so token content cannot abort it.
    std::string pending;
    double last_flush_ms = -1e9;
    for (int i = 0; ok && i < 512 && !stop_requested; ++i) {
        g_last_op.store(2); // sample
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) { stop_reason = "eog"; break; }
        g_last_op.store(3); // accept
        llama_sampler_accept(sampler, token);
        g_last_op.store(4); // to_piece
        std::vector<char> piece(256);
        int len = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        if (len < 0) { piece.resize(-len); len = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true); }
        if (len > 0) pending.append(piece.data(), (size_t) len);
        ++gen_tokens;
        g_gen_tokens.store(gen_tokens);
        double now_ms = ms_since(gen_start);
        // flush cut that never splits a multi-byte sequence
        size_t cut = pending.size();
        size_t j = cut; int back = 0;
        while (back < 4 && j > 0 && ((unsigned char) pending[j-1] & 0xC0) == 0x80) { --j; ++back; }
        if (j > 0) {
            unsigned char lead = (unsigned char) pending[j-1];
            size_t need = lead >= 0xF0 ? 4 : (lead >= 0xE0 ? 3 : (lead >= 0xC0 ? 2 : 1));
            if (need > 1 && cut - (j - 1) < need) cut = j - 1;
        } else cut = 0;
        if (cut > 0 && (cut >= 96 || gen_tokens == 1 || now_ms - last_flush_ms >= 100.0)) {
            std::string mu8;
            append_modified_utf8(mu8, pending.data(), cut);
            g_last_op.store(5); // jni_str
            jstring js = env->NewStringUTF(mu8.c_str());
            g_last_op.store(6); // jni_call
            if (js) { env->CallVoidMethod(callback, on_token, js); env->DeleteLocalRef(js); }
            g_last_op.store(7); // jni_exc
            if (env->ExceptionCheck()) { stop_reason = "error"; break; }
            pending.erase(0, cut);
            last_flush_ms = now_ms;
        }
        llama_batch batch = llama_batch_get_one(&token, 1);
        g_last_op.store(8); // gen_decode
        if (llama_decode(ctx, batch) != 0) { ok = false; stop_reason = "context_full"; }
        if ((gen_tokens & 127) == 0) { g_last_op.store(9); save_phase_state(); } // refresh native_state.txt every 128 tokens
    }
    if (!pending.empty() && !env->ExceptionCheck()) {
        std::string mu8;
        append_modified_utf8(mu8, pending.data(), pending.size());
        g_last_op.store(5); // jni_str
        jstring js = env->NewStringUTF(mu8.c_str());
        g_last_op.store(6); // jni_call
        if (js) { env->CallVoidMethod(callback, on_token, js); env->DeleteLocalRef(js); }
        g_last_op.store(7); // jni_exc
        env->ExceptionCheck(); // absorb status; final text must still reach the UI
        pending.clear();
    }
    if (stop_requested) stop_reason = "user_stop";
    double gen_ms = ms_since(gen_start);
    set_phase(7); // stats_build
    llama_sampler_free(sampler);

    if (on_stats && !env->ExceptionCheck()) {
        char buf[512];
        snprintf(buf, sizeof(buf),
            "{\"backend\":\"%s\",\"threads\":%d,\"ctx\":%d,\"load_ms\":%.1f,\"model_cached\":%d,\"gpu_fallback\":%d,"
            "\"prefill_tokens\":%d,\"prefill_ms\":%.1f,\"gen_tokens\":%d,\"gen_ms\":%.1f,\"stop\":\"%s\"}",
            backend.c_str(), cp.n_threads, (int) cp.n_ctx, loaded_now ? last_load_ms : 0.0,
            loaded_now ? 0 : 1, gpu_fallback ? 1 : 0, n, prefill_ms, gen_tokens, gen_ms, stop_reason.c_str());
        jstring js = env->NewStringUTF(buf);
        set_phase(8); // stats_callback — if the process dies here, Diag will show 'stats_callback'
        if (js) { env->CallVoidMethod(callback, on_stats, js); env->DeleteLocalRef(js); }
    }
    set_phase(9); // cleanup — llama_free (incl. Vulkan teardown) crashes surface with this phase
    llama_free(ctx);
    set_phase(10); // done
    if (!ok && !env->ExceptionCheck() && !stop_requested) fail(env, "Inference failed or context is full.");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pocketworkbench_app_NativeEngine_transcribe(JNIEnv * env, jobject, jstring path, jfloatArray samples, jstring log_dir) {
    std::string model_path = from_java(env, path);
    {
        std::string logs = from_java(env, log_dir);
        if (!logs.empty()) {
            snprintf(g_crash_file, sizeof(g_crash_file), "%s/native_crash.txt", logs.c_str());
            snprintf(g_state_file, sizeof(g_state_file), "%s/native_state.txt", logs.c_str());
            snprintf(g_model_hint, sizeof(g_model_hint), "speech");
            snprintf(g_backend_hint, sizeof(g_backend_hint), "whisper");
            static bool installed = false;
            if (!installed) { install_native_crash_handlers(); installed = true; }
        }
    }
    capture_thread_stack();
    set_phase(1);
    whisper_context_params context_params = whisper_context_default_params();
    whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), context_params);
    if (!ctx) { set_phase(10); fail(env, "Cannot open speech model."); return nullptr; }
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = std::clamp((int) std::thread::hardware_concurrency() - 1, 1, 8);
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.language = "auto";
    jsize len = env->GetArrayLength(samples);
    jfloat * pcm = env->GetFloatArrayElements(samples, nullptr);
    set_phase(6);
    int code = whisper_full(ctx, params, pcm, len);
    env->ReleaseFloatArrayElements(samples, pcm, JNI_ABORT);
    if (code != 0) { set_phase(9); whisper_free(ctx); set_phase(10); fail(env, "Speech transcription failed."); return nullptr; }
    std::string transcript;
    for (int i = 0; i < whisper_full_n_segments(ctx); ++i) transcript += whisper_full_get_segment_text(ctx, i);
    set_phase(9);
    whisper_free(ctx);
    set_phase(10);
    return env->NewStringUTF(transcript.c_str());
}

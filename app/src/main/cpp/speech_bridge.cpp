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
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <cstdint>
#include "whisper.h"

// Speech (whisper.cpp) JNI bridge. The LLM inference engine lives in Rust
// (rust/pocketinfer, libpocketinfer.so); this library only keeps whisper and
// the shared native crash forensics used by the app diagnostics.

static const char * kPhaseNames[] = {
    "idle", "load_model", "alloc_context", "chat_template", "tokenize",
    "prefill", "generating", "stats_build", "stats_callback", "cleanup", "done"
};
static const int kPhaseCount = (int) (sizeof(kPhaseNames) / sizeof(kPhaseNames[0]));
static std::atomic<int> g_phase{0};
static char g_crash_file[512] = {0};
static char g_state_file[512] = {0};

static void write_all(int fd, const char * s, size_t n);
static void write_str(int fd, const char * s);

static void write_all(int fd, const char * s, size_t n) {
    while (n > 0) { ssize_t k = write(fd, s, n); if (k <= 0) return; s += k; n -= (size_t) k; }
}
static void write_str(int fd, const char * s) { write_all(fd, s, strlen(s)); }

static void native_crash_handler(int sig, siginfo_t *, void *) {
    if (g_crash_file[0]) {
        int fd = open(g_crash_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            int p = g_phase.load();
            write_str(fd, "signal=");
            char b[24];
            snprintf(b, sizeof(b), "%d", sig);
            write_str(fd, b);
            write_str(fd, "\nphase=");
            write_str(fd, kPhaseNames[p >= 0 && p < kPhaseCount ? p : 0]);
            write_str(fd, "\nbackend=whisper\n");
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
    fprintf(f, "pid=%d\nphase=%s\nts=%lld\nbackend=whisper\n",
            (int) getpid(), kPhaseNames[p >= 0 && p < kPhaseCount ? p : 0],
            (long long) time(nullptr) * 1000LL);
    fclose(f);
}
static void set_phase(int p) { g_phase.store(p); save_phase_state(); }

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

extern "C" JNIEXPORT jstring JNICALL
Java_com_pocketworkbench_app_SpeechEngine_transcribe(JNIEnv * env, jobject, jstring path, jfloatArray samples, jstring log_dir) {
    std::string model_path = from_java(env, path);
    {
        std::string logs = from_java(env, log_dir);
        if (!logs.empty()) {
            snprintf(g_crash_file, sizeof(g_crash_file), "%s/native_crash.txt", logs.c_str());
            snprintf(g_state_file, sizeof(g_state_file), "%s/native_state.txt", logs.c_str());
            static bool installed = false;
            if (!installed) { install_native_crash_handlers(); installed = true; }
        }
    }
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

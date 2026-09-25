#include <jni.h>
#include <atomic>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include "llama.h"
#include "ggml-backend.h"
#include "whisper.h"

static std::atomic<bool> stop_requested{false};
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
extern "C" JNIEXPORT void JNICALL
Java_com_pocketworkbench_app_NativeEngine_stop(JNIEnv *, jobject) { stop_requested = true; }

extern "C" JNIEXPORT void JNICALL
Java_com_pocketworkbench_app_NativeEngine_generate(
        JNIEnv * env, jobject, jstring path, jobjectArray roles, jobjectArray contents, jobject callback) {
    stop_requested = false;
    std::string model_path = from_java(env, path);
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    const bool gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) || ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    mp.n_gpu_layers = gpu ? 99 : 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model && gpu) { mp.n_gpu_layers = 0; model = llama_model_load_from_file(model_path.c_str(), mp); }
    if (!model) { fail(env, "Cannot load GGUF model. Check format and free memory."); return; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_threads = std::clamp((int) std::thread::hardware_concurrency() - 1, 1, 8);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx && gpu && mp.n_gpu_layers > 0) {
        llama_model_free(model);
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(model_path.c_str(), mp);
        if (model) ctx = llama_init_from_model(model, cp);
    }
    if (!ctx) { if (model) llama_model_free(model); fail(env, "Cannot allocate model context."); return; }
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
    const char * tmpl = llama_model_chat_template(model, nullptr);
    int required = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, nullptr, 0);
    if (required <= 0 || required > 4 * 1024 * 1024) {
        llama_free(ctx); llama_model_free(model); fail(env, "Model chat template is unavailable."); return;
    }
    std::string prompt(required + 1, '\0');
    llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, prompt.data(), prompt.size());
    prompt.resize(required);
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n >= 0) { llama_free(ctx); llama_model_free(model); fail(env, "Tokenization failed."); return; }
    std::vector<llama_token> tokens(-n);
    n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (n <= 0 || n >= 3500) {
        llama_free(ctx); llama_model_free(model); fail(env, "Conversation is too long for the 4096-token context."); return;
    }
    tokens.resize(n);
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0xC0FFEE));
    jclass callback_type = env->GetObjectClass(callback);
    jmethodID on_token = env->GetMethodID(callback_type, "onToken", "(Ljava/lang/String;)V");
    bool ok = on_token != nullptr;
    for (int offset = 0; ok && offset < n && !stop_requested; offset += 512) {
        int count = std::min(512, n - offset);
        llama_batch batch = llama_batch_get_one(tokens.data() + offset, count);
        if (llama_decode(ctx, batch) != 0) ok = false;
    }
    for (int i = 0; ok && i < 512 && !stop_requested; ++i) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        llama_sampler_accept(sampler, token);
        std::vector<char> piece(256);
        int len = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        if (len < 0) { piece.resize(-len); len = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true); }
        if (len > 0) {
            std::string str(piece.data(), len);
            jstring js = env->NewStringUTF(str.c_str());
            if (js) { env->CallVoidMethod(callback, on_token, js); env->DeleteLocalRef(js); }
            if (env->ExceptionCheck()) break;
        }
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) ok = false;
    }
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    if (!ok && !env->ExceptionCheck() && !stop_requested) fail(env, "Inference failed or context is full.");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pocketworkbench_app_NativeEngine_transcribe(JNIEnv * env, jobject, jstring path, jfloatArray samples) {
    std::string model_path = from_java(env, path);
    whisper_context_params context_params = whisper_context_default_params();
    whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), context_params);
    if (!ctx) { fail(env, "Cannot open speech model."); return nullptr; }
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = std::clamp((int) std::thread::hardware_concurrency() - 1, 1, 8);
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.language = "auto";
    jsize len = env->GetArrayLength(samples);
    jfloat * pcm = env->GetFloatArrayElements(samples, nullptr);
    int code = whisper_full(ctx, params, pcm, len);
    env->ReleaseFloatArrayElements(samples, pcm, JNI_ABORT);
    if (code != 0) { whisper_free(ctx); fail(env, "Speech transcription failed."); return nullptr; }
    std::string transcript;
    for (int i = 0; i < whisper_full_n_segments(ctx); ++i) transcript += whisper_full_get_segment_text(ctx, i);
    whisper_free(ctx);
    return env->NewStringUTF(transcript.c_str());
}

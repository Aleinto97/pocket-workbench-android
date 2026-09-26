# Pocket Workbench — Android tablet source project

**Status: engineering preview, not a production-ready app.** This English-language Android app targets Snapdragon-class tablets. The v0.3.0 build replaces the llama.cpp LLM path with a **pure-Rust inference engine** (`rust/pocketinfer`, no external crates) with a CPU NEON/int8 backend, an experimental OpenCL GPU backend and Qualcomm QNN/Hexagon NPU detection. On-device validation of v0.3.0 was performed on the reference Snapdragon device; the APK built by CI still needs a fresh install/QA pass. Native dependencies are fetched during the build; no model or speech weights are bundled.

## Inference engine (v0.3.0)

- `rust/pocketinfer` — pure Rust, zero crates.io dependencies (only Rust `std` + raw FFI to Android libc/libOpenCL/QNN). Built as `libpocketinfer.so` by `scripts/build-rust.sh`; Gradle runs the script before `preBuild` and stages the library into `app/build/rustjni/arm64-v8a`.
- **GGUF loader**: metadata v2/v3, tensor table, mmap zero-copy weights, dequant for F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K/Q2_K/Q3_K.
- **Tokenizer**: exact GPT-2 byte-level BPE with the `minicpm5` (and `gpt-2`, `qwen2`, `llama3`) pre-tokenizer splits, special-token parsing, `MiniCPM5` chat template and the documented direct-answer (`enable_thinking=false`) prefix.
- **CPU backend**: aarch64 NEON kernels, an int8 `dotprod` path for Q4_K/Q6_K/Q4_0/Q8_0 matmuls, persistent worker pool. Best measured with 4 threads on the reference device.
- **GPU backend (experimental)**: Adreno OpenCL via `dlopen` (runtime-compiled kernels, no external shader toolchain). It is used only when the user selects GPU and silently falls back to CPU on any error; it has not been device-verified yet.
- **NPU**: the engine detects a QNN/Hexagon runtime and a `.qnn` context binary next to the GGUF, reports it in Stats, and keeps CPU/GPU until on-device QNN execution lands (see roadmap).
- **Memory safety net**: Rust-side signal forensics still writes `native_state.txt`/`native_crash.txt` (phase, op, token counts, backend) so the existing Diagnostics screen keeps working.

### Measured on the reference Snapdragon device (MiniCPM5-2B-Q4_K_M, 4 threads)

- Tokenizer parity with llama.cpp: identical token ids on the test corpus (including digit chunking and `minicpm5` pre-tokenizer).
- Greedy generation parity with llama.cpp on the real MiniCPM5-2B GGUF: 16/16 identical tokens.
- Decode ≈ 2.8–3.0 tok/s, prefill ≈ 3.8 tok/s after the int8 activation path (llama.cpp CPU on the same device/thermal state: 7.5 tok/s decode, 12.3 tok/s prefill). Further gains require i8mm batched GEMM and the OpenCL/NPU paths.

## Implemented

- Adaptive tablet navigation with conversation sidebar on wide windows and a bottom bar on compact windows. Chat keeps its scroll position while older messages are read; reasoning and tool results can be expanded independently. Individual messages and full conversations can be copied. Conversation and model deletion require confirmation.
- Hugging Face GGUF repository search, model list, import, resumable `.part` downloads, storage checks, progress, and basic GGUF signature validation. Downloads and chat history are kept in app-private storage. A failed transfer can be resumed by tapping Download again.
- Streaming token callback through the Rust JNI bridge, multi-turn chat template, selectable 4K/8K/16K context (8K default), local history, Stop, GPU request or CPU selection. MiniCPM5 offers a direct-answer option, selected by default; the engine applies its documented no-thinking prefix only when the template matches, otherwise uses automatic reasoning. The app records the actual backend and falls back to CPU on backend discovery, model-load or context-allocation failures. The last model stays resident; the prompt is re-processed for each agent step. Completed reasoning stays visible in chat but is excluded from subsequent inference prompts. When the prompt is too large, oldest turns are removed from the inference input by exact token count while the UI history remains intact.
- **Technical performance log:** every generation records backend (CPU/OpenCL GPU, with GPU-fallback detection), threads, model load time, prefill and generation tok/s, stop reason, RAM state, and device identity. Runs are shown as a tok/s chip under each assistant reply and listed in the **Stats** page with detected problems (context overflow, inference errors, low memory, low speed, GPU fallback), exportable as a Markdown report.
- **Local agent and GitHub integration:** Agent mode is enabled by default and can list, read and write files in its app-private workspace and run Android shell commands without GitHub sign-in. OAuth Device Flow adds GitHub repository, issue, pull request and Actions tools when signed in (token stored with Android Keystore). The parser accepts both JSON `<tool>` and MiniCPM5's XML `<function>` calls.
- Microphone capture at 16 kHz and `whisper.cpp` on-device speech recognition (unchanged: speech stays in the C++ bridge, `libpocketnative.so`). Install the `ggml-base.bin` speech model from the Models page, speak, review the transcribed text, and press Send.
- A Files screen for browsing the agent's workspace, previewing text files, exporting individual files and folders as ZIP through Android's document picker.

## Gaps against the requested finished app

- **NPU execution is not implemented yet.** The engine can detect the QNN runtime and a context binary, but running MiniCPM5 on the Hexagon NPU requires exporting the model with Qualcomm AI Hub / QAIRT into a QNN context binary and binding the QNN C API on device. Until then the NPU row in Stats is informational only.
- **OpenCL GPU path is unverified on hardware.** It compiles, is behind the existing GPU toggle, and any failure falls back to CPU with `gpu_fallback=1` in Stats.
- **Throughput is below llama.cpp's tuned CPU/Vulkan kernels** (see measurements above): i8mm batched GEMM, async OpenCL pipelining and QNN offload are the roadmap.
- **Agent tool loop is not device-tested in this revision.** Small GGUF models may generate malformed calls or stop before writing complete projects. Each turn has a 12-step limit; a command has a 90-second timeout. There is no in-app web preview.
- Downloads are bound to the app process. Android may terminate them if it kills the process; `.part` files remain for retry. There is no background foreground-service notification, cancellation button, SHA-256 verification against Hub metadata, model quality ranking, or automatic split GGUF handling.
- A release build should also include profiling, error and memory tests, accessibility checks, background transfer tests, and a privacy/security review.

## Build on a machine with Android Studio

1. Install Android Studio, Android SDK platform 35, NDK (27.2.12479018) and CMake 3.22.1. Use Java 17 and Gradle 8.11.1 with Android Gradle Plugin 8.9.2.
2. Install Rust (rustup) and the Android target: `rustup target add aarch64-linux-android`. The engine is pure `std`, so no crates are downloaded.
3. From the project root run `./scripts/fetch-native.sh` (whisper.cpp only in v0.3.0); record the printed commit.
4. Build with `ANDROID_NDK_HOME=<ndk> bash scripts/build-rust.sh` (Gradle also runs it automatically before `preBuild`), then `gradle :app:assembleDebug` or build from Android Studio. A CPU-only build works out of the box; OpenCL is discovered at runtime.
5. Install the APK with `adb install app/build/outputs/apk/debug/app-debug.apk`. Download a small single-file, un-gated instruction GGUF, choose it in Chat, and test offline with Wi-Fi off. Download the speech model before offline testing.

## Notes for the device

Prefer a small Q4 GGUF first (for example 1–3B parameters); MiniCPM5-2B-Q4_K_M is the validated configuration. A 32B GGUF can exceed practical Android app memory even when the file fits storage. Repository licenses and chat templates vary and must be checked by the user.

Privacy: model discovery and downloads connect to Hugging Face; GitHub sign-in and agent tools connect to GitHub when enabled. Inference, transcript generation, and history are on device. Model publishers can see download requests, and GitHub receives requests made through its tools. The app requests microphone permission only when voice capture starts. History and downloaded models are excluded from Android backup.

## Key modules

- `app/src/main/java/com/pocketworkbench/app/NativeEngine.kt`: Rust engine JNI surface (generate/stop).
- `app/src/main/java/com/pocketworkbench/app/SpeechEngine.kt`: whisper.cpp transcription.
- `app/src/main/java/com/pocketworkbench/app/MainActivity.kt`: tablet UI and microphone/document pickers.
- `app/src/main/java/com/pocketworkbench/app/WorkbenchViewModel.kt`: chat, recording, transfers, shell.
- `rust/pocketinfer/src/`: engine (gguf, quant, tokenizer, model, backends, JNI, forensics).
- `app/src/main/cpp/speech_bridge.cpp`: whisper.cpp JNI integration.

## Development validation

- `python3 scripts/check-project.py` verifies project structure, manifest permissions, JNI method correspondence and file naming without an Android SDK.
- `cd rust/pocketinfer && cargo test` runs quantization/kernel layout tests, including int8 NEON vs f32 comparisons.
- `python3 rust/pocketinfer/tools/make_tiny_model.py tiny-f32.gguf` generates a tiny MiniCPM5-shaped GGUF for engine debugging (`cargo run --bin pocketinfer-cli`).

## Optional remote builds

`.gitlab-ci.yml` builds a debug APK on GitLab CI, runs the Rust tests, cross-compiles the engine, and publishes APK + symbols to the package registry. `.github/workflows/android-build.yml` mirrors the build on GitHub Actions. CI success is not tablet validation or a production release.

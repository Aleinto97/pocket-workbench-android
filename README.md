# Pocket Workbench — Android tablet source project

**Status: engineering preview.** This Android app targets Snapdragon tablets, including the SM8845P reference device. LLM inference uses a Rust engine with no external Rust crates. CPU inference works; OpenCL is experimental. **NPU inference is not implemented**: QNN/FastRPC information in Stats is diagnostic only. The APK bundles neither model weights nor a QNN runtime.

## Inference engine

- `rust/pocketinfer` — Rust with zero crates.io dependencies (Rust `std` + raw FFI to Android libc/OpenCL; runtime probing uses `dlopen`). Built as `libpocketinfer.so` by `scripts/build-rust.sh`; Gradle runs the script before `preBuild` and stages the library into `app/build/rustjni/arm64-v8a`.
- **GGUF loader**: metadata v2/v3, tensor table, mmap zero-copy weights, dequant for F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K/Q2_K/Q3_K.
- **Tokenizer**: exact GPT-2 byte-level BPE with the `minicpm5` (and `gpt-2`, `qwen2`, `llama3`) pre-tokenizer splits, special-token parsing, `MiniCPM5` chat template and the documented direct-answer (`enable_thinking=false`) prefix.
- **CPU backend**: NEON-accelerated single-token attention and RMSNorm; NEON Q8 activation quantization for decode, dot-product Q4_K/Q6_K/Q4_0/Q8_0 kernels, and a worker pool with row prefetch and a participating caller thread. `f32` KV improves access speed but doubles KV RAM compared with f16. RoPE and SwiGLU remain scalar; there is no i8mm/SME kernel.
- **GPU backend (experimental, off by default)**: Adreno OpenCL via `dlopen` (runtime-compiled kernels, no external shader toolchain). The toggle is off after the first install; when enabled it must pass an on-device self-test against the CPU kernels, and any failure falls back to CPU automatically. A native crash on the GPU path sets a `gpu_safe_mode` marker that keeps OpenCL off until the user re-arms it.
- **NPU diagnostics**: reports whether a QNN context file or runtime can be seen; neither is sufficient to execute MiniCPM5. There is no QNN tensor I/O, graph execution, KV-state integration, or hardware validation. Stats never labels CPU tokens as NPU tokens.
- **Crash diagnostics**: the regular state snapshot records the model/backend; the Android signal handler uses only fixed buffers, atomics and async-signal-safe libc calls to record the signal, phase, operation, backend and token counts. Rust panics are caught at the JNI boundary.
- **Statistics page**: native library load status, engine capabilities (NEON/int8 dotprod/OpenCL/QNN/FastRPC), loaded model details, a one-tap **health check** (file → GGUF → tokenizer → engine load → prefill → sampling) that reports the exact failing step, engine errors recorded in the run history, plus the previous crash/diagnostics log.
- **Statistics during generation**: `engineInfo` returns `busy` immediately while inference holds the model (fixes the v0.3.4 ANR).
- **Inference reliability**: Stop interrupts prefill between layers, callback errors end generation, and scratch buffers survive cancellation. Model load checks tensor shapes against configuration and tokenizer, refuses missing output normalization, and uses checked tensor offsets and sizes. Norm and attention buffers scale with model dimensions; logits are sized from the actual output tensor.

### Validation on the SM8845P reference device (MiniCPM5-2B-Q4_K_M, GGUF metadata 2.6B)

The Rust CLI produced the same first 16 greedy token IDs as llama.cpp for the raw prompt `The capital of France is`. This is one corpus check, not a proof of numeric parity for every prompt or context. Timings observed in short CLI runs depend on context size, thread count, CPU temperature, memory pressure and warmup; a matched, repeated on-device benchmark is needed before claiming a speedup over llama.cpp or v0.3.5.

For this model the f32 KV cache uses approximately 336 / 672 / 1344 MiB at 4K / 8K / 16K context (f16 used half that). Choose context size with available Android RAM in mind.

- Tokenizer parity with llama.cpp: identical token ids on the test corpus (including digit chunking and `minicpm5` pre-tokenizer).
- In one raw-prompt check, 16/16 first greedy token IDs matched llama.cpp (`The capital of France is` → ` Paris.\n- The capital of Germany is`).
- The OpenCL GPU backend runs a **self-test** (synthetic Q4_K/Q6_K matvec compared against the CPU kernel) before it is used; if the kernels do not match on the device the engine falls back to CPU and Stats reports `gpu_fallback`.

## Implemented

- Adaptive tablet navigation with conversation sidebar on wide windows and a bottom bar on compact windows. Chat keeps its scroll position while older messages are read; reasoning and tool results can be expanded independently. Individual messages and full conversations can be copied. Conversation and model deletion require confirmation.
- Hugging Face GGUF repository search, model list, import, resumable `.part` downloads, storage checks, progress, and basic GGUF signature validation. Downloads and chat history are kept in app-private storage. A failed transfer can be resumed by tapping Download again.
- Streaming token callback through the Rust JNI bridge, multi-turn chat template, selectable 4K/8K/16K context (8K default), local history, Stop, GPU request or CPU selection. MiniCPM5 offers a direct-answer option, selected by default; the engine applies its documented no-thinking prefix only when the template matches, otherwise uses automatic reasoning. The app records the actual backend and falls back to CPU on backend discovery, model-load or context-allocation failures. The last model stays resident; the prompt is re-processed for each agent step. Completed reasoning stays visible in chat but is excluded from subsequent inference prompts. When the prompt is too large, oldest turns are removed from the inference input by exact token count while the UI history remains intact.
- **Technical performance log:** every generation records backend (CPU/OpenCL GPU, with GPU-fallback detection), threads, model load time, prefill and generation tok/s, stop reason, RAM state, and device identity. Runs are shown as a tok/s chip under each assistant reply and listed in the **Stats** page with detected problems (context overflow, inference errors, low memory, low speed, GPU fallback), exportable as a Markdown report.
- **Local agent and GitHub integration:** Agent mode is enabled by default and can list, read and write files in its app-private workspace and run Android shell commands without GitHub sign-in. OAuth Device Flow adds GitHub repository, issue, pull request and Actions tools when signed in (token stored with Android Keystore). The parser accepts both JSON `<tool>` and MiniCPM5's XML `<function>` calls.
- Microphone capture at 16 kHz and `whisper.cpp` on-device speech recognition (unchanged: speech stays in the C++ bridge, `libpocketnative.so`). Install the `ggml-base.bin` speech model from the Models page, speak, review the transcribed text, and press Send.
- A Files screen for browsing the agent's workspace, previewing text files, exporting individual files and folders as ZIP through Android's document picker.

## Gaps against the requested finished app

- **NPU execution is not implemented.** The current GGUF Q4_K_M file cannot be passed directly to QNN. An actual NPU backend needs an export compatible with MiniCPM5, QAIRT runtime and version-matched headers, named graph and tensor binding, prefill/decode with persistent KV state, and numerical/performance validation on this tablet. The device's vendor libraries can be inaccessible to Android app processes.
- **OpenCL GPU path is unverified on hardware.** It compiles, is behind the existing GPU toggle, and any failure falls back to CPU with `gpu_fallback=1` in Stats.
- **Performance versus llama.cpp is not established under controlled conditions.** Further CPU work could include i8mm batched GEMM, lower-overhead long-context attention and matched thermal/memory benchmarks; the OpenCL path also needs real-model validation.
- **Agent tool loop is not device-tested in this revision.** Small GGUF models may generate malformed calls or stop before writing complete projects. Each turn has a 12-step limit; a command has a 90-second timeout. There is no in-app web preview.
- Downloads are bound to the app process. Android may terminate them if it kills the process; `.part` files remain for retry. There is no background foreground-service notification, cancellation button, SHA-256 verification against Hub metadata, model quality ranking, or automatic split GGUF handling.
- A release build should also include profiling, error and memory tests, accessibility checks, background transfer tests, and a privacy/security review.

## Path to Hexagon NPU inference

The QNN row in Stats is informational; this release always generates LLM tokens on CPU or experimental OpenCL. To develop QNN support, first obtain the licensed QAIRT SDK/runtime and a MiniCPM5-compatible model export for the device, including an explicit prefill/decode graph interface and KV-cache state. Then bind the SDK's *actual versioned* API from Rust, pass correctly described tensors, and compare logits and streamed tokens against the CPU implementation on hardware before enabling or reporting `NPU`. Merely placing a `.qnn.bin` alongside a Q4_K_M GGUF does not perform a conversion or offload.

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

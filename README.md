# Pocket Workbench — Android tablet source project

**Status: engineering preview, not a production-ready app.** This English-language, landscape-friendly Android app is designed around HONOR MagicPad 4 (Snapdragon 8 Gen 5). A CPU-only ARM64 debug APK was built successfully by GitHub Actions on September 25, 2026 ([run #6](https://github.com/Aleinto97/pocket-workbench-android/actions/runs/36124794064)); it has not been run on the tablet. Native dependencies are fetched during the build; no proprietary model or speech weights are bundled.

## Implemented

- Tablet layout with conversation navigation, model library, performance stats, and workspace; compact layout has a history dialog.
- Hugging Face GGUF repository search, model list, import, resumable `.part` downloads, storage checks, progress, and basic GGUF signature validation. Downloads and chat history are kept in app-private storage. A failed transfer can be resumed by tapping Download again.
- `llama.cpp` JNI streaming token callback, multi-turn chat template, 4096-token context, local history, Stop, CPU inference and optional Vulkan GPU build with load-time CPU fallback. The last model stays resident in memory, so repeated generations (including agent tool loops) skip the load phase.
- **Technical performance log:** every generation records backend (CPU/Vulkan, with GPU-fallback detection), threads, model load time, prefill and generation tok/s, stop reason, RAM state, and device identity (SoC, cores, Android version). Runs are shown as a tok/s chip under each assistant reply, listed in a dedicated **Stats** page with detected problems (context overflow, inference errors, low memory, low speed, GPU fallback) and exportable as a Markdown report via copy or share.
- **GitHub integration:** OAuth Device Flow sign-in (user enters a one-time code at github.com/login/device; the token is encrypted at rest with the Android Keystore). A configurable OAuth App Client ID is required once. With **Agent mode** enabled, the on-device model can call a built-in MCP-style tool set: list/read repositories and files, code search, issues and comments, branch/file commits, pull requests, and GitHub Actions (list workflows and runs, inspect job logs, dispatch a build). Tool calls and results appear inline in chat. A "repo workflow" token scope is required for Actions tools.
- Microphone capture at 16 kHz and `whisper.cpp` on-device speech recognition. Install the `ggml-base.bin` speech model from the Models page, speak, review/edit the resulting text, and press Send. The model download is online once; transcription is offline afterward.
- An app-private workspace and shell console for manually running Android shell commands and inspecting/editing files.

## Gaps against the requested finished app

- **No integrated Linux distribution, Docker-like container, or package manager.** The Workspace tab uses `/system/bin/sh`, which runs in the Android app's context. It must not be presented as isolated Linux. A real userland would need a separately packaged and tested PRoot/Alpine integration, compatible executable delivery, and device testing; Android 10+ restricts executing downloaded binaries from writable app storage.
- **Agent tool loop is not device-tested.** The MCP-style GitHub tools rely on the local model reliably emitting `<tool>` JSON; small GGUF models may produce malformed calls (the app feeds an error back and retries once per step). NPU remains unsupported; Actions tools require a "repo workflow" scoped token.
- **No validated NPU support.** Upstream Hexagon paths are device-specific. CPU is the default; Vulkan build is optional and untested on this tablet. The GPU fallback covers load/context creation, not driver failure mid-generation.
- Downloads are bound to the app process. Android may terminate them if it kills the process; `.part` files remain for retry. There is no background foreground-service notification, cancellation button, SHA-256 verification against Hub metadata, model quality ranking, or automatic split GGUF handling.
- The Android debug build passed CI, but no device test has been performed. Native upstream branches must be pinned and API compatibility verified before release. A release build should also include profiling, error and memory tests, accessibility checks, background transfer tests, and a privacy/security review.

## Build on a machine with Android Studio

1. Install Android Studio, Android SDK platform 35, NDK and CMake 3.22.1. Use Java 17 and Gradle 8.11.1 compatible with Android Gradle Plugin 8.9.2.
2. From the project root run `./scripts/fetch-native.sh`; record the printed commits. The `native/` source checkouts are intentionally excluded from this source archive, so network access is required once for setup. Inspect the upstream licenses before redistribution.
3. Open the project in Android Studio. If it asks for a Gradle wrapper, use a system Gradle 8.11.1 to run `gradle wrapper --gradle-version 8.11.1`, then sync and run `:app:assembleDebug`.
4. For an experimental Vulkan APK, use `./gradlew :app:assembleDebug -Pgpu=true` after installing NDK Vulkan tooling including `glslc` if requested by CMake. Start with the CPU build and test the GPU variant separately on the MagicPad 4.
5. Install the APK with Android Studio or `adb install app/build/outputs/apk/debug/app-debug.apk`. Download a small single-file, un-gated instruction GGUF, choose it in Chat, and test offline with Wi-Fi off. Download the speech model before offline testing.

## Notes for the device

Prefer a small Q4 GGUF first (for example 1–3B parameters). Try 7–8B Q4 only after confirming the tablet's actual RAM and free storage. A 32B GGUF can exceed practical Android app memory even when the file fits storage. The model browser returns compatible-looking GGUF files, but repository licenses and model chat templates vary and must be checked by the user.

Privacy: the app talks to `huggingface.co` only for model discovery and downloads; inference, transcript generation, and history are on device. The model publisher can see download requests. The app requests microphone permission only when voice capture starts. History and downloaded models are excluded from Android backup.

## Key modules

- `app/src/main/java/com/pocketworkbench/app/MainActivity.kt`: tablet UI and microphone/document pickers.
- `app/src/main/java/com/pocketworkbench/app/WorkbenchViewModel.kt`: chat, recording, transfers, shell.
- `app/src/main/java/com/pocketworkbench/app/Data.kt`: Hub client, resumable transfer, local persistence.
- `app/src/main/cpp/bridge.cpp`: llama.cpp and whisper.cpp JNI integration.

## Development validation

The included `scripts/check-project.py` verifies project structure, manifest permissions, JNI method correspondence, required states, and file naming constraints without an Android SDK. It is a static smoke check only; a passing result does not mean the project compiles or works on a device.

## Optional remote build

`.github/workflows/android-build.yml` builds a debug APK on GitHub Actions and uploads it as a workflow artifact. [Run #6](https://github.com/Aleinto97/pocket-workbench-android/actions/runs/36124794064) passed. A successful CI build alone is not tablet validation or a production release.

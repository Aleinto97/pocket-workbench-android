# Pocket Workbench — Android tablet source project

**Status: engineering preview, not a production-ready app.** This English-language Android app targets the HONOR tablet. The v0.2.9 build includes Vulkan with CPU fallback. A v0.2.7 on-device run confirmed Vulkan GPU generation on HONOR YLE-W09 (1749 tokens at 25.2 tok/s) but reached critical thermal status; the local workspace agent and file exports still need device tests. Native dependencies are fetched during the build; no model or speech weights are bundled.

## Implemented

- Adaptive tablet navigation with conversation sidebar on wide windows and a bottom bar on compact windows. Chat keeps its scroll position while older messages are read; reasoning and tool results can be expanded independently. Individual messages and full conversations can be copied. Conversation and model deletion require confirmation.
- Hugging Face GGUF repository search, model list, import, resumable `.part` downloads, storage checks, progress, and basic GGUF signature validation. Downloads and chat history are kept in app-private storage. A failed transfer can be resumed by tapping Download again.
- `llama.cpp` JNI streaming token callback, multi-turn chat template, selectable 4K/8K/16K context (8K default), local history, Stop, Vulkan GPU request or CPU selection. MiniCPM5 offers a direct-answer option, selected by default; the native code applies its documented no-thinking prefix only when the template matches, otherwise uses automatic reasoning. GitLab/GitHub CI build with Vulkan; the app records the actual backend and falls back to CPU on GPU discovery, model-load or context-allocation failures. The last model stays resident, but the prompt is still processed again for each agent step. Completed reasoning stays visible in chat but is excluded from subsequent inference prompts. When the prompt is too large, oldest turns are removed from the inference input by exact token count while the UI history remains intact.
- **Technical performance log:** every generation records backend (CPU/Vulkan, with GPU-fallback detection), threads, model load time, prefill and generation tok/s, stop reason, RAM state, and device identity (SoC, cores, Android version). Runs are shown as a tok/s chip under each assistant reply, listed in a dedicated **Stats** page with detected problems (context overflow, inference errors, low memory, low speed, GPU fallback) and exportable as a Markdown report via copy or share.
- **Local agent and GitHub integration:** Agent mode is enabled by default and can list, read and write files in its app-private workspace and run Android shell commands without GitHub sign-in. OAuth Device Flow adds GitHub repository, issue, pull request and Actions tools when signed in (token stored with Android Keystore). The parser accepts both JSON `<tool>` and MiniCPM5's XML `<function>` calls. Tool results appear inline in chat. A "repo workflow" token scope is required for Actions tools.
- Microphone capture at 16 kHz and `whisper.cpp` on-device speech recognition. Install the `ggml-base.bin` speech model from the Models page, speak, review/edit the resulting text, and press Send. The model download is online once; transcription is offline afterward.
- A Files screen for browsing the agent's workspace, previewing text files, exporting individual files and exporting folders as ZIP through Android's document picker. Shell commands are only available to the agent; no command-entry UI is shown.

## Gaps against the requested finished app

- **No integrated Linux distribution, Docker-like container, or package manager.** The agent's command tool uses `/system/bin/sh` under the Android app's UID. Installed commands depend on the device, and downloaded executables in writable app storage face Android restrictions. Packaging a real PRoot/Alpine userland and verifying builds on the device remains separate work.
- **Agent tool loop is not device-tested.** Small GGUF models may generate malformed calls or stop before writing complete projects. Each turn has a 12-step limit; a command has a 90-second timeout. A project can contain files even if no runnable build tools are available. There is no in-app web preview. NPU remains unsupported; Actions tools require a "repo workflow" scoped token.
- **No validated NPU support.** Vulkan was observed on the HONOR tablet in v0.2.7, but direct-answer mode and the agent loop remain untested there. The CPU fallback covers backend discovery, model loading and context creation; a driver crash during generation cannot be recovered in-process. The larger context consumes more memory and the agent still reprocesses its prompt after every tool call.
- Downloads are bound to the app process. Android may terminate them if it kills the process; `.part` files remain for retry. There is no background foreground-service notification, cancellation button, SHA-256 verification against Hub metadata, model quality ranking, or automatic split GGUF handling.
- The Android debug build passed CI, but no device test has been performed. Native upstream branches must be pinned and API compatibility verified before release. A release build should also include profiling, error and memory tests, accessibility checks, background transfer tests, and a privacy/security review.

## Build on a machine with Android Studio

1. Install Android Studio, Android SDK platform 35, NDK and CMake 3.22.1. Use Java 17 and Gradle 8.11.1 compatible with Android Gradle Plugin 8.9.2.
2. From the project root run `./scripts/fetch-native.sh`; record the printed commits. The `native/` source checkouts are intentionally excluded from this source archive, so network access is required once for setup. Inspect the upstream licenses before redistribution.
3. Open the project in Android Studio. If it asks for a Gradle wrapper, use a system Gradle 8.11.1 to run `gradle wrapper --gradle-version 8.11.1`, then sync and run `:app:assembleDebug`.
4. To match CI, build with `gradle :app:assembleDebug -Pgpu=true` after installing Vulkan development headers, SPIR-V headers and `glslc`. Choose CPU or Vulkan on the Models page and check the measured backend in Stats. A CPU-only build can still be made without `-Pgpu=true`.
5. Install the APK with Android Studio or `adb install app/build/outputs/apk/debug/app-debug.apk`. Download a small single-file, un-gated instruction GGUF, choose it in Chat, and test offline with Wi-Fi off. Download the speech model before offline testing.

## Notes for the device

Prefer a small Q4 GGUF first (for example 1–3B parameters). Try 7–8B Q4 only after confirming the tablet's actual RAM and free storage. A 32B GGUF can exceed practical Android app memory even when the file fits storage. The model browser returns compatible-looking GGUF files, but repository licenses and model chat templates vary and must be checked by the user.

Privacy: model discovery and downloads connect to Hugging Face; GitHub sign-in and agent tools connect to GitHub when enabled. Inference, transcript generation, and history are on device. Model publishers can see download requests, and GitHub receives requests made through its tools. The app requests microphone permission only when voice capture starts. History and downloaded models are excluded from Android backup.

## Key modules

- `app/src/main/java/com/pocketworkbench/app/MainActivity.kt`: tablet UI and microphone/document pickers.
- `app/src/main/java/com/pocketworkbench/app/WorkbenchViewModel.kt`: chat, recording, transfers, shell.
- `app/src/main/java/com/pocketworkbench/app/Data.kt`: Hub client, resumable transfer, local persistence.
- `app/src/main/cpp/bridge.cpp`: llama.cpp and whisper.cpp JNI integration.

## Development validation

The included `scripts/check-project.py` verifies project structure, manifest permissions, JNI method correspondence, required states, and file naming constraints without an Android SDK. It is a static smoke check only; a passing result does not mean the project compiles or works on a device.

## Optional remote build

`.github/workflows/android-build.yml` builds a debug APK on GitHub Actions and uploads it as a workflow artifact. [Run #6](https://github.com/Aleinto97/pocket-workbench-android/actions/runs/36124794064) passed. A successful CI build alone is not tablet validation or a production release.

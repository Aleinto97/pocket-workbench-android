# NPU Bench

Measures whether the Hexagon NPU is worth using for Pocket Workbench, by
comparing it against the CPU on the same device with the same model.

## Why this is an app and not a terminal command

The ggml-hexagon backend asks RPCCode for an absolute skel URI, hardcoded in
`ggml-hexagon.cpp`:

    file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0

RPCCode resolves that relative to the calling process's native library
directory. From `adb shell` there is none, and the skel cannot be placed where
it would be found, because the device has no root and `/vendor` is read-only.
The session then fails with `error 0x80000406`. The libraries therefore have to
be installed by the package manager, which only happens for an app.

## Layout

- `src/main/jniLibs/arm64-v8a/` — the GenieX / llama.cpp runtime, including
  `libggml-htp-v81.so`, the DSP-side skel for this chip. Not committed; see
  `scripts/fetch-geniex.sh`.
- `src/main/assets/geniex-bench` — the benchmark executable. It is an ELF
  program rather than a shared library, so it cannot live in jniLibs, which
  Android only unpacks for `lib*.so`.
- `src/main/java/.../MainActivity.kt` — runs the benchmark once for `--device npu`
  and once for `--device cpu`, and prints both.

## Usage

    bash scripts/fetch-geniex.sh          # needs HF_TOKEN
    gradle :npubench:assembleRelease
    adb install -r npubench/build/outputs/apk/release/npubench-release.apk
    adb push qwen3-0.6b-q4_0.gguf /data/data/com.pocketworkbench.npubench/files/bench.gguf
    adb shell am start -n com.pocketworkbench.npubench/.MainActivity

The model must be `Q4_0`: that is the quantisation the Hexagon backend is
built for, and it is why a `Q4_K_M` model does not land on the NPU.

## Reading the result

Compare `pp512` (prefill) and `tg32` (generation) between the two runs. The
baseline to beat is the app's own engine on MiniCPM5-2B: 12.4 tok/s generation
and about 16.7 tok/s prefill, measured on this device under `severe` thermal
throttling. If the NPU does not clear that, the NPU is not worth the 100 MB of
runtime in the APK.

All numbers are unthrottled by thermal state, so treat them as relative
comparisons rather than absolute performance.

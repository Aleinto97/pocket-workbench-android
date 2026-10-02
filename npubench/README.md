# NPU Bench

Measures whether the Hexagon NPU is worth using for Pocket Workbench, by
comparing it against the CPU on the same device with the same model.

## Why this is an app and not a terminal command

The ggml-hexagon backend asks RPCCode for an absolute skel URI, hardcoded in
`ggml-hexagon.cpp`:

    file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0

The GenieX plugin sets `ADSP_LIBRARY_PATH` to `GENIEX_PLUGIN_PATH`, where RPCCode
looks for the skel. The app links the skel from its installed native library
directory into that path. From `adb shell` there is no app library directory,
and the device has no root to install the skel under `/vendor`. The session
then fails with `error 0x80000406`.

## Layout

- `src/main/jniLibs/arm64-v8a/` — the GenieX / llama.cpp runtime, including
  `libggml-htp-v81.so`, the DSP-side skel for this chip, and the benchmark
  executable packaged as `libgeniexbench.so`. Android extracts `lib*.so` into
  the app's executable native library directory. Not committed; see
  `scripts/fetch-geniex.sh`.
- `src/main/c/bench_exit.c` — compiles into an LD_PRELOAD shim that flushes the
  completed report and exits the one-shot process before GenieX unloads its
  plugin. Requires the Android NDK or an Android arm64 C compiler when fetching.
- `src/main/java/.../MainActivity.kt` — runs the benchmark once for `--device npu`
  and once for `--device cpu`, and prints both. It creates a child-directory
  link for GenieX's plugin scanner and a skel link at `GENIEX_PLUGIN_PATH`
  because the plugin sets `ADSP_LIBRARY_PATH` to that directory.

## Usage

    bash scripts/fetch-geniex.sh          # needs HF_TOKEN or a cached tarball
    gradle :npubench:assembleDebug
    adb install -r npubench/build/outputs/apk/debug/npubench-debug.apk
    adb shell mkdir -p /data/local/tmp/npb
    adb push qwen3-0.6b-q4_0.gguf /data/local/tmp/npb/bench.gguf
    adb shell am start -n com.pocketworkbench.npubench/.MainActivity

Set `ANDROID_NDK_HOME` to the installed NDK before running the fetch script on
a workstation; in Termux, the `aarch64-linux-android-clang` command is also
supported.

This Hexagon build can offload both `Q4_0` and `Q4_K_M` weights, depending on
the model and its operations. Check `HTP0 model buffer size` against `CPU model
buffer size` in `bench-npu.log`: a successful HTP session or a reported number
of offloaded layers alone does not prove that the weights ran on the NPU.
Qwen3.8-27B GSQ-RCO IQ2_XS, for example, kept nearly 8 GiB of weights in CPU.

### Larger local models and token generation

Public GGUFs tested on the 16 GB tablet:

- [Qwen3.8-4B Distill Q4_K_M](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF/blob/main/Qwen3.8-4B-Q4_K_M.gguf)
  (2.78 GB): 2.81 GiB in HTP0, 0.50 GiB in CPU; 14.9 tok/s NPU decode versus
  6.4 tok/s CPU in the 512/32 speed test. The NPU also generated a tool-call
  shaped JSON response to an Italian instruction.
- [Qwen3-14B Q4_K_M](https://huggingface.co/unsloth/Qwen3-14B-GGUF/blob/main/Qwen3-14B-Q4_K_M.gguf)
  (9.0 GB): CPU, highest-capacity tested; memory gets tight while generating.
- [Qwen3-8B Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf)
  (5.0 GB): CPU, more memory headroom.
- [Qwen3-4B Q4_0](https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_0.gguf)
  (2.4 GB): NPU, fastest larger model tested.

After downloading and pushing a GGUF under `/data/local/tmp/npb/`, select it
by filename. For example, to verify *actual text generation* with the 14B:

    adb push Qwen3-14B-Q4_K_M.gguf /data/local/tmp/npb/qwen3-14b-q4km.gguf
    adb shell am force-stop com.pocketworkbench.npubench
    adb shell am start -n com.pocketworkbench.npubench/.MainActivity \
      --es model qwen3-14b-q4km.gguf --es device cpu --ez accuracy true

Accuracy mode uses the model's chat template and a real Italian question,
prints the generated text (`[gen ]`) and counts up to 96 generated tokens by
default (`--ei tokens 24` selects a shorter run). Use `--es prompt "your question"`
to choose another prompt. Omitting `device`
runs both devices on the same model; omit `accuracy` for the 512-token prefill
and 32-token decode speed comparison.

## Reading the result

Compare prefill and decode token rates between the two runs. Reports are saved
as `files/bench-npu.json` and `files/bench-cpu.json` in the app's private data;
the complete native output is retained in `files/bench-npu.log` and
`files/bench-cpu.log`. Each benchmark process has a 120-second timeout.
On a debug build, read the app log with
`adb shell run-as com.pocketworkbench.npubench cat files/npubench.log`.

The baseline to beat is the app's own engine on MiniCPM5-2B: 12.4 tok/s generation
and about 16.7 tok/s prefill, measured on this device under `severe` thermal
throttling. If the NPU does not clear that, the NPU is not worth the 100 MB of
runtime in the APK.

Measurements are not normalized for thermal state, so treat them as relative
comparisons rather than absolute performance.

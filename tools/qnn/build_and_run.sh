#!/usr/bin/env bash
# Verified QNN/HTP pipeline for the Snapdragon 8 Elite Gen 5 (Hexagon V81).
#
# What this does, end to end, with only a Linux aarch64 host and the phone:
#   1. ONNX -> DLC           with the SDK's Python converter
#   2. DLC -> V81 context    prepared ON THE DEVICE (qairt-dlc-prepare)
#   3. context binary -> NPU via FastRPC, checked against a numpy reference
#
# Why step 2 runs on the device: qnn-context-binary-generator wants a QNN
# network .so produced by qnn-model-lib-generator, and that tool does not ship
# for Linux aarch64. qairt-dlc-prepare takes the DLC directly and runs fine
# from adb shell.
#
# Requirements:
#   - QAIRT SDK 2.50.0.260828, unzipped so that $QNN_SDK_ROOT/qairt/<ver> exists
#   - Python 3.12 (the converter refuses 3.13 on aarch64) with numpy, onnx,
#     pyyaml, flatbuffers, setuptools (for the removed distutils) and packaging
#   - phone connected over adb
set -euo pipefail

QNN_SDK_ROOT="${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the unzipped SDK/qairt/<version> dir}"
PY="${PY:?set PY to a python3.12 interpreter}"
ONNX="${1:?usage: build_and_run.sh model.onnx [input.raw]}"
INPUT_RAW="${2:-}"

ANDROID_LIB="$QNN_SDK_ROOT/lib/aarch64-android"
ANDROID_BIN="$QNN_SDK_ROOT/bin/aarch64-android"
SKEL="$QNN_SDK_ROOT/lib/hexagon-v81/unsigned"
REMOTE=/data/local/tmp/qnn

echo "== 1. ONNX -> DLC =="
PYTHONPATH="$QNN_SDK_ROOT/lib/python" "$PY" "$(dirname "$0")/onnx_qnn_shim.py" \
  "$ANDROID_BIN/qairt-converter" \
  --input_network "$ONNX" --output_path model.dlc --target_backend HTP \
  ${DIMS:-}

echo "== 2. DLC -> V81 context binary on the device =="
adb shell mkdir -p "$REMOTE"
adb push model.dlc "$ANDROID_BIN/qairt-dlc-prepare" "$ANDROID_BIN/qairt-net-run" "$REMOTE/" >/dev/null
for f in libQairtHtp.so libQairtHtpPrepare.so libQairtSystem.so \
         libQairtHtpV81Stub.so libQnnModelDlc.so; do
  adb push "$ANDROID_LIB/$f" "$REMOTE/" >/dev/null
done
for f in libQairtHtpV81.so libQairtHtpV81Skel.so; do
  adb push "$SKEL/$f" "$REMOTE/" >/dev/null
done
adb shell "chmod 755 $REMOTE/*.so $REMOTE/qairt-* "
adb shell "cd $REMOTE && ADSP_LIBRARY_PATH='$REMOTE;/vendor/lib/rfsa/adsp;/dsp' \
  LD_LIBRARY_PATH=$REMOTE:/vendor/lib64 \
  ./qairt-dlc-prepare --backend libQairtHtp.so --input_dlc model.dlc \
    --binary_file model_v81.bin --output_dir $REMOTE/out"

echo "== 3. run on the HTP =="
if [ -n "$INPUT_RAW" ]; then
  adb push "$INPUT_RAW" "$REMOTE/" >/dev/null
  printf 'x:=%s/%s\n' "$REMOTE" "$(basename "$INPUT_RAW")" > input_list.txt
  adb push input_list.txt "$REMOTE/" >/dev/null
  LIST="--input_list $REMOTE/input_list.txt --num_inferences 1"
else
  LIST=""
fi
adb shell "cd $REMOTE && ADSP_LIBRARY_PATH='$REMOTE;/vendor/lib/rfsa/adsp;/dsp' \
  LD_LIBRARY_PATH=$REMOTE:/vendor/lib64 \
  ./qairt-net-run --input_context_binary $REMOTE/out/model_v81.bin \
    --backend libQairtHtp.so $LIST --output_dir $REMOTE/runout"

echo "outputs in $REMOTE/runout"

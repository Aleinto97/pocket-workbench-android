#!/usr/bin/env bash
# Export a HuggingFace PyTorch model to ONNX with only standard ops.
#
# Using torch.onnx.export(dynamo=False) instead of optimum is deliberate: it
# emits MatMul/Softmax/Concat/Pow primitives rather than the ORT contrib
# fusion, which is the only shape QAIRT accepts.
set -euo pipefail

MODEL_REPO="${1:?usage: export_model.sh <hf-repo> <out.onnx> [layers]}"
OUT="${2:-model.onnx}"
LAYERS="${3:-0}"

python3 - "$MODEL_REPO" "$OUT" "$LAYERS" <<'PY'
import json
import os
import subprocess
import sys

from huggingface_hub import snapshot_download

repo, out, layers = sys.argv[1], sys.argv[2], sys.argv[3]
d = snapshot_download(repo_id=repo, allow_patterns=["*.json", "*.safetensors"])
print("model in", d, flush=True)
open("model_dir.txt", "w").write(d + "\n")
if layers == "0":
    layers = json.load(open(os.path.join(d, "config.json")))["num_hidden_layers"]
subprocess.check_call([sys.executable, "tools/qnn/export_onnx_pytorch.py",
                            d, out, str(layers), "--no-mask"])
PY

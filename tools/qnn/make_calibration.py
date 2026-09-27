import os
import re
import sys
import numpy as np
import onnx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if os.path.isfile(sys.argv[1]):
    MODEL_DIR = open(sys.argv[1]).read().strip()
else:
    MODEL_DIR = sys.argv[1]
OUTDIR, NSAMPLES = sys.argv[2], int(sys.argv[3])
NO_MASK = "--no-mask" in sys.argv
B, Q, PAST = 1, 1, 127
os.makedirs(OUTDIR, exist_ok=True)

# Take the tensor names from the exported graph instead of assuming them: the
# quantizer's netrun binds calibration files by name, and the DLC may have been
# renamed during conversion.
def qnn_name(name):
    """QAIRT rewrites ONNX tensor names into its own identifier charset:
    the converter turns "past_key.1" into "past_key_1". The quantizer's netrun
    binds calibration files by the DLC name, so emit that form."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


onnx_path = os.environ.get("ONNX_MODEL", "model.onnx")
graph_inputs = [i.name for i in onnx.load(onnx_path, load_external_data=False).graph.input]
print("tensori dal grafo:", len(graph_inputs))

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).eval()
cfg = model.config
LAYERS = cfg.num_hidden_layers

prompt = ("The capital of France is Paris. The capital of Germany is Berlin. "
          "The capital of Italy is Rome. The capital of Spain is Madrid. "
          "Explain what a neural network is, in a few sentences.")
ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + ids
ids = (ids + [tok.eos_token_id or 0] * (PAST + Q))[:PAST + Q]
print("token di contesto:", len(ids))

# prefill 127 token, poi un passo di decode: la cache risultante e' una
# calibrazione realistica per il grafo decode (seq=1, past=127)
with torch.no_grad():
    pre = model(input_ids=torch.tensor([ids[:PAST]]), use_cache=True)
    step = model(input_ids=torch.tensor([ids[PAST:PAST + Q]]),
                 past_key_values=pre.past_key_values, use_cache=True)

nkv, hd = cfg.num_key_value_heads, getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

for s in range(NSAMPLES):
    last = ids[PAST - 1 + s % 1] if s == 0 else ids[PAST + (s % (len(ids) - PAST))]
    files = {
        "input_ids": np.array([[last]], dtype=np.int64),
        "position_ids": np.array([[PAST]], dtype=np.int64),
    }
    if not NO_MASK:
        files["attention_mask"] = np.ones((B, PAST + Q), dtype=np.int64)
    # assign the exported names positionally: the legacy exporter numbers most
    # KV inputs ("past_key.1") but leaves the last pair unnumbered, so deriving
    # the name arithmetically is wrong.
    kv_inputs = [n for n in graph_inputs if n not in ("input_ids", "position_ids", "attention_mask")]
    if len(kv_inputs) != 2 * LAYERS:
        raise SystemExit("expected %d KV inputs, graph has %d" % (2 * LAYERS, len(kv_inputs)))
    for l in range(LAYERS):
        k = step.past_key_values.layers[l].keys[0].to(torch.float16).numpy()
        v = step.past_key_values.layers[l].values[0].to(torch.float16).numpy()
        name_k, name_v = kv_inputs[2 * l], kv_inputs[2 * l + 1]
        if "key" not in name_k or "value" not in name_v:
            raise SystemExit("unexpected KV input order: %s, %s" % (name_k, name_v))
        files[name_k] = k
        files[name_v] = v
    missing = [n for n in graph_inputs if n not in files]
    extra = [n for n in files if n not in graph_inputs]
    if missing or extra:
        raise SystemExit("calibration does not match the graph: missing=%s extra=%s"
                         % (missing[:4], extra[:4]))
    lines = []
    for name, arr in files.items():
        fn = f"s{s}_{qnn_name(name)}.raw"
        arr.tofile(os.path.join(OUTDIR, fn))
        lines.append(f"{qnn_name(name)}:={fn}")
    with open(os.path.join(OUTDIR, f"input_list_{s}.txt"), "w") as f:
        f.write(" ".join(lines) + "\n")
print("campioni scritti in", OUTDIR, "| tensori per campione:", len(files))

import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR, OUTDIR, NSAMPLES = sys.argv[1], sys.argv[2], int(sys.argv[3])
B, Q, PAST = 1, 1, 127
os.makedirs(OUTDIR, exist_ok=True)

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
        "attention_mask": np.ones((B, PAST + Q), dtype=np.int64),
        "position_ids": np.array([[PAST]], dtype=np.int64),
    }
    for l in range(LAYERS):
        k, v = step.past_key_values.layers[l].keys, step.past_key_values.layers[l].values
        tag = 2 * l + 1
        files[f"past_key.{tag}"] = k[0].to(torch.float16).numpy()
        files[f"past_value.{tag}"] = v[0].to(torch.float16).numpy()
    lines = []
    for name, arr in files.items():
        fn = f"s{s}_{name}.raw"
        arr.tofile(os.path.join(OUTDIR, fn))
        lines.append(f"{name}:={fn}")
    with open(os.path.join(OUTDIR, f"input_list_{s}.txt"), "w") as f:
        f.write(" ".join(lines) + "\n")
print("campioni scritti in", OUTDIR, "| tensori per campione:", len(files))

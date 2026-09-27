import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_DIR = sys.argv[1]
OUT = sys.argv[2]
LAYERS = int(sys.argv[3])
# For a decode graph (q_len == 1) the causal mask lets the new token see the
# whole cache, so the padding-mask branch is dead weight. Dropping it also drops
# the boolean ops (And/Where) that QNN HTP rejects.
NO_MASK = "--no-mask" in sys.argv
KEEP = sys.argv[4] if len(sys.argv) > 4 else "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj lm_head"

# Force eager attention: the ONNX export of scaled_dot_product_attention
# emulates safe-softmax with IsNaN/Where, which QNN rejects at op validation.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, attn_implementation="eager")
model.eval()
cfg = model.config
print("layers", cfg.num_hidden_layers, "kv heads", cfg.num_key_value_heads,
      "head_dim", getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
      "hidden", cfg.hidden_size)

B, Q, PAST = 1, 1, 127
dtype = torch.float16


class Wrapper(nn.Module):
    def __init__(self, m, n_layers):
        super().__init__()
        self.m = m
        self.n = n_layers

    def forward(self, *all_args):
        if NO_MASK:
            input_ids, position_ids = all_args[0], all_args[1]
            attention_mask, past = None, all_args[2:]
        else:
            input_ids, attention_mask, position_ids = all_args[0], all_args[1], all_args[2]
            past = all_args[3:]
        cache = DynamicCache()
        for i in range(self.n):
            cache.update(past[2 * i], past[2 * i + 1], i)
        out = self.m(input_ids=input_ids, attention_mask=attention_mask,
                     position_ids=position_ids, past_key_values=cache, use_cache=True)
        presents = []
        for layer in out.past_key_values.layers[:self.n]:
            presents.append(layer.keys)
            presents.append(layer.values)
        return (out.logits, *presents)


w = Wrapper(model, LAYERS)
w.eval()

names = ["input_ids", "attention_mask", "position_ids"]
args = [torch.zeros((B, Q), dtype=torch.long),
        torch.zeros((B, PAST + Q), dtype=torch.long),
        torch.zeros((B, Q), dtype=torch.long)]
if NO_MASK:
    names.remove("attention_mask")
    del args[1]
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
nkv = cfg.num_key_value_heads
for _ in range(LAYERS):
    names += ["past_key", "past_value"]
    args += [torch.zeros((B, nkv, PAST, hd), dtype=dtype),
             torch.zeros((B, nkv, PAST, hd), dtype=dtype)]

outs = ["logits"] + [f"present_{i}_{t}" for i in range(LAYERS) for t in ("k", "v")]

print("esporto con torch.onnx.export legacy, opset 14 ...")
torch.onnx.export(
    w,
    tuple(args),
    OUT,
    input_names=names,
    output_names=outs,
    opset_version=14,
    do_constant_folding=True,
    dynamo=False,
    export_params=True,
)
print("scritto", OUT)

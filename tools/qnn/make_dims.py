"""Extract static input dims for the QAIRT converter and reject fused graphs.

QAIRT cannot consume ORT-fused contrib ops (GroupQueryAttention,
MatMulNBits, SkipSimplifiedLayerNormalization, RotaryEmbedding), which is
what optimum/onnx-community exports emit. Failing here is much cheaper than
letting the converter die halfway through a multi-minute run.
"""
import sys

import onnx

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "dims.txt"

m = onnx.load(path, load_external_data=False)
nodes = list(m.graph.node)
domains = {n.domain for n in nodes}
fused = sorted(d for d in domains if d)
if fused:
    counts = {}
    for n in nodes:
        if n.domain:
            counts[n.op_type] = counts.get(n.op_type, 0) + 1
    raise SystemExit("fused ONNX domains present, not convertible: %s (%s)" % (fused, counts))

dims = " ".join(
    "-d %s %s" % (i.name, ",".join(str(int(x.dim_value)) for x in i.type.tensor_type.shape.dim))
    for i in m.graph.input
)
with open(out, "w") as f:
    f.write(dims + "\n")

print("nodes", len(nodes), "| inputs", len(m.graph.input), "| outputs", len(m.graph.output))
print("domains: standard only")
print("dims written to", out)

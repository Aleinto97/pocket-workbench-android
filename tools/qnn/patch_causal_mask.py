import sys
import collections
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
B, Q, KV = 1, 1, 512

m = onnx.load(src, load_external_data=False)
g = m.graph
nodes = list(g.node)
initnames = {i.name for i in g.initializer}
ginputnames = {i.name for i in g.input}
gi_types = {vi.name: vi.type.tensor_type.elem_type for vi in list(g.input) + list(g.value_info)}

prod = {}
for k, n in enumerate(nodes):
    for o in n.output:
        prod[o] = k

tri_idx = next(k for k, n in enumerate(nodes) if n.op_type == "Trilu")
trilu = nodes[tri_idx]

# backward closure over node indices
mask_set = set()
stack = [tri_idx]
while stack:
    k = stack.pop()
    if k in mask_set:
        continue
    mask_set.add(k)
    for i in nodes[k].input:
        if i in initnames or i not in prod:
            continue
        stack.append(prod[i])
print("mask subgraph nodes:", len(mask_set),
      sorted({nodes[k].op_type for k in mask_set}))

# eval_set is the closure to evaluate; remove_set drops it again. Trilu itself
# must go: the folded Constant takes over its output, which the outside Mul reads.
outside = {i for k, n in enumerate(nodes) if k not in mask_set for i in n.input}
keep = set()
stack = [k for k in mask_set
         if k != tri_idx and any(o in outside for o in nodes[k].output)]
keep.update(stack)
# retention must be transitive: keeping a node keeps its mask ancestors alive
while stack:
    k = stack.pop()
    for i in nodes[k].input:
        p = prod.get(i)
        if p in mask_set and p not in keep:
            keep.add(p)
            stack.append(p)
remove_set = mask_set - keep
print("mask nodes evaluated:", len(mask_set), "| removed:", len(remove_set))

# initializers the subgraph needs, to evaluate it standalone
need_init = {i for k in mask_set for i in nodes[k].input if i in initnames}
ginputs = sorted({i for k in mask_set for i in nodes[k].input
                  if i in ginputnames})
print("subgraph inputs:", ginputs)

sub_nodes = [nodes[k] for k in sorted(mask_set)]
sub = helper.make_graph(
    sub_nodes,
    "mask",
    [helper.make_tensor_value_info(i, gi_types.get(i, TensorProto.INT64), None) for i in ginputs],
    [helper.make_tensor_value_info(trilu.output[0],
                                   gi_types.get(trilu.output[0], TensorProto.FLOAT16), None)],
    [i for i in g.initializer if i.name in need_init],
)
sub_model = helper.make_model(sub, opset_imports=m.opset_import)
sub_model.ir_version = 13
onnx.save(sub_model, "_mask_sub.onnx")

sess = ort.InferenceSession("_mask_sub.onnx", providers=["CPUExecutionProvider"])
feed = {"attention_mask": np.zeros((B, KV), dtype=np.int64),
        "input_ids": np.zeros((B, Q), dtype=np.int64)}
mask = sess.run(None, feed)[0]
print("evaluated mask", mask.shape, mask.dtype)

folded = [n for k, n in enumerate(nodes) if k not in remove_set]
folded.insert(0, helper.make_node(
    "Constant", [], [trilu.output[0]], name="folded_causal_mask",
    value=numpy_helper.from_array(mask, trilu.output[0])))

# topological sort by index
produced = {}
for k, n in enumerate(folded):
    for o in n.output:
        produced[o] = k
indeg = [0] * len(folded)
dependents = collections.defaultdict(list)
for k, n in enumerate(folded):
    for i in n.input:
        if i in produced:
            indeg[k] += 1
            dependents[produced[i]].append(k)
queue = collections.deque(k for k in range(len(folded)) if indeg[k] == 0)
order = []
while queue:
    k = queue.popleft()
    order.append(k)
    for d in dependents.get(k, []):
        indeg[d] -= 1
        if indeg[d] == 0:
            queue.append(d)
if len(order) != len(folded):
    raise SystemExit("unresolved: %d of %d" % (len(folded) - len(order), len(folded)))

copies = []
for k in order:
    c = onnx.NodeProto()
    c.CopyFrom(folded[k])
    copies.append(c)
del g.node[:]
g.node.extend(copies)

used = {i for n in g.node for i in n.input}
keep_init = [i for i in g.initializer if i.name in used]
del g.initializer[:]
g.initializer.extend(keep_init)

onnx.checker.check_model(m, full_check=False)
onnx.save(m, dst)
print("saved", dst, "| nodes", len(g.node), "| init", len(g.initializer),
      "| Trilu gone:", not any(n.op_type == "Trilu" for n in g.node))

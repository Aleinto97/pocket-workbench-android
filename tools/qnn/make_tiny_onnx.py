import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

rng = np.random.default_rng(7)
I, H, O = 64, 128, 32

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, I])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, O])

w1n = np.ascontiguousarray(rng.standard_normal((I, H)).astype(np.float32) * 0.1)
w1 = numpy_helper.from_array(w1n, "w1")
b1n = rng.standard_normal(H).astype(np.float32) * 0.1
b1 = numpy_helper.from_array(b1n, "b1")
w2n = np.ascontiguousarray(rng.standard_normal((H, O)).astype(np.float32) * 0.1)
w2 = numpy_helper.from_array(w2n, "w2")
b2n = rng.standard_normal(O).astype(np.float32) * 0.1
b2 = numpy_helper.from_array(b2n, "b2")

nodes = [
    helper.make_node("MatMul", ["x", "w1"], ["h1"], name="fc1"),
    helper.make_node("Add", ["h1", "b1"], ["h1b"], name="fc1_bias"),
    helper.make_node("Relu", ["h1b"], ["a1"], name="relu1"),
    helper.make_node("MatMul", ["a1", "w2"], ["h2"], name="fc2"),
    helper.make_node("Add", ["h2", "b2"], ["y"], name="fc2_bias"),
]

g = helper.make_graph(nodes, "tiny", [x], [y], [w1, b1, w2, b2])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)])
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, "tiny.onnx")

# reference output for correctness comparison on device
xv = rng.standard_normal((1, I)).astype(np.float32)
h1 = np.maximum(xv @ w1n + b1n, 0)
ref = h1 @ w2n + b2n
xv.tofile("tiny_in.raw")
ref.tofile("tiny_ref.raw")
print("tiny.onnx creato; riferimento salvato", ref.shape)

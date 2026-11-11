"""Hand-built policy/value model (no training): material value + capture-biased policy."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Plane order: Ping, Daemon, Exploit, Trace, Ice, Kernel (own 0-5, enemy 6-11).
MATERIAL = np.array([15, 45, 25, 30, 50, 0], dtype=np.float32)

# Value weights over the flattened (16*8*8) observation.
w_val = np.zeros((16, 8, 8), dtype=np.float32)
centre = np.array([[min(r, 7 - r, c, 7 - c) for c in range(8)] for r in range(8)], dtype=np.float32)
for p in range(6):
    w_val[p] = MATERIAL[p] + 1.5 * centre        # own pieces, small centralisation bonus
    w_val[6 + p] = -(MATERIAL[p] + 1.5 * centre)
w_val[5] += 4.0 * centre                          # keep own Kernel away from the firewall
w_val[11] -= 4.0 * centre
w_val = (w_val / 40.0).reshape(1024, 1)

# Policy: logit bonus for landing on a tile holding an enemy piece, scaled by its value.
w_pol = np.zeros((16, 64, 64), dtype=np.float32)
for p in range(6):
    v = 100.0 if p == 5 else MATERIAL[p]
    for s in range(64):
        w_pol[6 + p, s, s] = v / 15.0
w_pol = w_pol.reshape(1024, 64)

inits = [
    numpy_helper.from_array(w_val, "w_val"),
    numpy_helper.from_array(w_pol, "w_pol"),
    numpy_helper.from_array(np.zeros((1, 64, 64), dtype=np.float32), "zeros"),
    numpy_helper.from_array(np.array([-1, 1024], dtype=np.int64), "flat_shape"),
    numpy_helper.from_array(np.array([-1, 4096], dtype=np.int64), "pol_shape"),
    numpy_helper.from_array(np.array([-1], dtype=np.int64), "val_shape"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
]
nodes = [
    helper.make_node("Reshape", ["observation", "flat_shape"], ["flat"]),
    helper.make_node("MatMul", ["flat", "w_val"], ["v_raw"]),
    helper.make_node("Tanh", ["v_raw"], ["v_tanh"]),
    helper.make_node("Reshape", ["v_tanh", "val_shape"], ["value"]),
    helper.make_node("MatMul", ["flat", "w_pol"], ["to_bonus"]),
    helper.make_node("Unsqueeze", ["to_bonus", "ax1"], ["to_b3"]),
    helper.make_node("Add", ["to_b3", "zeros"], ["pol3"]),
    helper.make_node("Reshape", ["pol3", "pol_shape"], ["policy"]),
]
graph = helper.make_graph(
    nodes, "colony_material",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "model.onnx")
print("saved model.onnx")

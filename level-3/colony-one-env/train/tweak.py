"""Post-hoc tweak of a merged model: policy logits *= P, value pre-tanh *= V.

    python train/tweak.py in.onnx out.onnx P V
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst, P, V = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
m = onnx.load(src); g = m.graph
tanh = None
for nd in g.node:
    if "policy" in nd.output:
        nd.output[:] = ["policy_raw" if o == "policy" else o for o in nd.output]
    if nd.op_type == "Tanh" and "value" in nd.output:
        tanh = nd
pre = tanh.input[0]
g.node.remove(tanh)
g.initializer.extend([numpy_helper.from_array(np.float32(P), "tw_p"), numpy_helper.from_array(np.float32(V), "tw_v")])
g.node.append(helper.make_node("Mul", ["policy_raw", "tw_p"], ["policy"]))
g.node.append(helper.make_node("Mul", [pre, "tw_v"], ["value_pre_scaled"]))
g.node.append(helper.make_node("Tanh", ["value_pre_scaled"], ["value"]))
onnx.checker.check_model(m)
onnx.save(m, dst)
print("saved", dst, P, V)

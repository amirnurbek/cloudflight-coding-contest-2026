"""Hand-built Level 4 model (no training): material + threat maps + centralisation.

Threat maps come from fixed convolutions: one kernel per piece type marking the tiles it
can hit next action (blockers ignored). The side to move gains part of the best enemy
piece it attacks and loses part of the best own piece the enemy attacks.

    python build_model4.py out.onnx [gain] [loss] [centre] [kcentre] [scale]
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

out = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
GAIN, LOSS, CENTRE, KCENTRE, SCALE = (float(a) for a in (sys.argv[2:7] + ["0.6", "0.3", "1.5", "4.0", "60"][len(sys.argv[2:7]):]))

# Plane order: Ping, Daemon, Exploit, Trace, Ice, Kernel (own 0-5, enemy 6-11).
MATERIAL = np.array([15, 45, 25, 30, 50, 0], dtype=np.float32)
THREAT_VALUE = np.array([15, 45, 25, 30, 50, 100], dtype=np.float32)


def attack_kernel(p):
    k = np.zeros((7, 7), dtype=np.float32)
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            ad, ar, ac = max(abs(dr), abs(dc)), abs(dr), abs(dc)
            diag, orth = ar == ac and ar > 0, (dr == 0) != (dc == 0)
            if p == 0 and diag and ar <= 2: k[dr + 3, dc + 3] = 1            # Ping: diagonal 1-2
            if p == 1 and ((ad == 1) or (orth and ad in (2, 3))): k[dr + 3, dc + 3] = 1  # Daemon
            if p == 2 and {ar, ac} == {1, 2}: k[dr + 3, dc + 3] = 1          # Exploit: L-jump
            if p == 3 and diag and ar in (2, 3): k[dr + 3, dc + 3] = 1       # Trace: diagonal 2-3
            if p == 4 and (diag or orth) and ad <= 2: k[dr + 3, dc + 3] = 1  # Ice: 8 dirs up to 2
            if p == 5 and ad == 1: k[dr + 3, dc + 3] = 1                     # Kernel: adjacent
    return k


centre = np.array([[min(r, 7 - r, c, 7 - c) for c in range(8)] for r in range(8)], dtype=np.float32)

# Linear part over the flattened observation: material + centralisation.
w_lin = np.zeros((16, 8, 8), dtype=np.float32)
for p in range(6):
    w_lin[p] = MATERIAL[p] + CENTRE * centre
    w_lin[6 + p] = -(MATERIAL[p] + CENTRE * centre)
w_lin[5] += KCENTRE * centre
w_lin[11] -= KCENTRE * centre
w_lin = w_lin.reshape(1024, 1)

# Conv: channel 0 = tiles own pieces attack, channel 1 = tiles enemy pieces attack,
# channel 2 = value of own piece on tile, channel 3 = value of enemy piece on tile.
w_conv = np.zeros((4, 16, 7, 7), dtype=np.float32)
for p in range(6):
    w_conv[0, p] = attack_kernel(p)
    w_conv[1, 6 + p] = attack_kernel(p)
    w_conv[2, p, 3, 3] = THREAT_VALUE[p]
    w_conv[3, 6 + p, 3, 3] = THREAT_VALUE[p]

# Policy: logit bonus for landing on an enemy piece, scaled by its value.
w_pol = np.zeros((16, 64, 64), dtype=np.float32)
for p in range(6):
    for s in range(64):
        w_pol[6 + p, s, s] = THREAT_VALUE[p] / 15.0
w_pol = w_pol.reshape(1024, 64)

c = lambda a, n: numpy_helper.from_array(np.asarray(a), n)
inits = [
    c(w_lin, "w_lin"), c(w_conv, "w_conv"), c(w_pol, "w_pol"),
    c(np.zeros((1, 64, 64), dtype=np.float32), "zeros"),
    c(np.array([-1, 1024], dtype=np.int64), "flat_shape"),
    c(np.array([-1, 4096], dtype=np.int64), "pol_shape"),
    c(np.array([-1], dtype=np.int64), "val_shape"),
    c(np.array([1], dtype=np.int64), "ax1"),
    c(np.array([1, 2, 3], dtype=np.int64), "red_axes"),
    c(np.float32(0.0), "zero_f"), c(np.float32(1.0), "one_f"),
    c(np.float32(GAIN), "gain"), c(np.float32(LOSS), "loss"), c(np.float32(1.0 / SCALE), "inv_scale"),
]
nodes = [
    helper.make_node("Reshape", ["observation", "flat_shape"], ["flat"]),
    helper.make_node("MatMul", ["flat", "w_lin"], ["lin2"]),
    helper.make_node("Reshape", ["lin2", "val_shape"], ["lin"]),
    helper.make_node("Conv", ["observation", "w_conv"], ["maps"], pads=[3, 3, 3, 3]),
    helper.make_node("Split", ["maps"], ["own_att_r", "enemy_att_r", "own_val", "enemy_val"], axis=1, num_outputs=4),
    helper.make_node("Clip", ["own_att_r", "zero_f", "one_f"], ["own_att"]),
    helper.make_node("Clip", ["enemy_att_r", "zero_f", "one_f"], ["enemy_att"]),
    helper.make_node("Mul", ["own_att", "enemy_val"], ["hit_enemy"]),
    helper.make_node("Mul", ["enemy_att", "own_val"], ["hit_own"]),
    helper.make_node("ReduceMax", ["hit_enemy", "red_axes"], ["best_gain"], keepdims=0),
    helper.make_node("ReduceMax", ["hit_own", "red_axes"], ["worst_loss"], keepdims=0),
    helper.make_node("Mul", ["best_gain", "gain"], ["g"]),
    helper.make_node("Mul", ["worst_loss", "loss"], ["l"]),
    helper.make_node("Add", ["lin", "g"], ["v1"]),
    helper.make_node("Sub", ["v1", "l"], ["v2"]),
    helper.make_node("Mul", ["v2", "inv_scale"], ["v3"]),
    helper.make_node("Tanh", ["v3"], ["value"]),
    helper.make_node("MatMul", ["flat", "w_pol"], ["to_bonus"]),
    helper.make_node("Unsqueeze", ["to_bonus", "ax1"], ["to_b3"]),
    helper.make_node("Add", ["to_b3", "zeros"], ["pol3"]),
    helper.make_node("Reshape", ["pol3", "pol_shape"], ["policy"]),
]
graph = helper.make_graph(
    nodes, "colony_threat",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out)
print("saved", out, dict(gain=GAIN, loss=LOSS, centre=CENTRE, kcentre=KCENTRE, scale=SCALE))

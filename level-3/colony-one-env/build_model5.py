"""Hand-built Level 4 model v5: v4 (material, 1-action threats, centre) plus
2-action reach on Kernels, Kernel guards, and the firewall ring.

    python build_model5.py out.onnx [key=value ...]
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

P = dict(gain=0.8, loss=0.2, centre=0.5, kcentre=4.0, scale=40.0,
         ku=60.0, kg=15.0, kb=60.0, fw=80.0)
out = sys.argv[1]
for kv in sys.argv[2:]:
    k, v = kv.split("="); P[k] = float(v)

MATERIAL = np.array([15, 45, 25, 30, 50, 0], dtype=np.float32)
THREAT_VALUE = np.array([15, 45, 25, 30, 50, 100], dtype=np.float32)


def pattern(fn):
    k = np.zeros((7, 7), dtype=np.float32)
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            ar, ac = abs(dr), abs(dc)
            if fn(ar, ac, max(ar, ac), ar == ac and ar > 0, (dr == 0) != (dc == 0)):
                k[dr + 3, dc + 3] = 1
    return k


ATTACK = [
    pattern(lambda ar, ac, ad, dg, og: dg and ar <= 2),                       # Ping
    pattern(lambda ar, ac, ad, dg, og: ad == 1 or (og and ad in (2, 3))),     # Daemon
    pattern(lambda ar, ac, ad, dg, og: {ar, ac} == {1, 2}),                   # Exploit
    pattern(lambda ar, ac, ad, dg, og: dg and ar in (2, 3)),                  # Trace
    pattern(lambda ar, ac, ad, dg, og: (dg or og) and ad <= 2),               # Ice
    pattern(lambda ar, ac, ad, dg, og: ad == 1),                              # Kernel
]
MOVE = [  # where the piece can stand after one move (including staying)
    pattern(lambda ar, ac, ad, dg, og: ad == 0 or (og and ad <= 2)),
    pattern(lambda ar, ac, ad, dg, og: ad == 0 or (og and ad <= 3)),
    pattern(lambda ar, ac, ad, dg, og: ad == 0 or {ar, ac} == {1, 2}),
    pattern(lambda ar, ac, ad, dg, og: ad == 0 or (og and ad == 1)),
    pattern(lambda ar, ac, ad, dg, og: ad == 0 or ((dg or og) and ad <= 2)),
    pattern(lambda ar, ac, ad, dg, og: ad <= 1),
]
centre = np.array([[min(r, 7 - r, c, 7 - c) for c in range(8)] for r in range(8)], dtype=np.float32)

w_lin = np.zeros((16, 8, 8), dtype=np.float32)
for p in range(6):
    w_lin[p] = MATERIAL[p] + P["centre"] * centre
    w_lin[6 + p] = -(MATERIAL[p] + P["centre"] * centre)
w_lin[5] += P["kcentre"] * centre
w_lin[11] -= P["kcentre"] * centre

w_conv = np.zeros((4, 16, 7, 7), dtype=np.float32)   # own att, enemy att, own val, enemy val
w_move = np.zeros((12, 16, 7, 7), dtype=np.float32)  # positions after one move, per type
w_att2 = np.zeros((2, 12, 7, 7), dtype=np.float32)   # 2-action reach: own, enemy
w_guard = np.zeros((2, 16, 3, 3), dtype=np.float32)  # adjacent non-Kernel friends
for p in range(6):
    w_conv[0, p] = ATTACK[p]; w_conv[1, 6 + p] = ATTACK[p]
    w_conv[2, p, 3, 3] = THREAT_VALUE[p]; w_conv[3, 6 + p, 3, 3] = THREAT_VALUE[p]
    w_move[p, p] = MOVE[p]; w_move[6 + p, 6 + p] = MOVE[p]
    w_att2[0, p] = ATTACK[p]; w_att2[1, 6 + p] = ATTACK[p]
    if p < 5:
        w_guard[0, p] = 1; w_guard[1, 6 + p] = 1
        w_guard[0, p, 1, 1] = 0; w_guard[1, 6 + p, 1, 1] = 0
# 1x1 picks: own kernel, enemy kernel, locked, actions_left, urgency, side-to-move
w_pick = np.zeros((6, 16, 1, 1), dtype=np.float32)
for i, plane in enumerate((5, 11, 12, 13, 14, 15)):
    w_pick[i, plane] = 1
edge = (centre == 0).astype(np.float32)[None, None]

w_pol = np.zeros((16, 64, 64), dtype=np.float32)
for p in range(6):
    for s in range(64):
        w_pol[6 + p, s, s] = THREAT_VALUE[p] / 15.0

inits, nodes = [], []


def const(name, a):
    inits.append(numpy_helper.from_array(np.asarray(a), name)); return name


def node(op, ins, out, **kw):
    nodes.append(helper.make_node(op, ins, [out] if isinstance(out, str) else out, **kw)); return out


f = lambda x: np.float32(x)
const("w_lin", w_lin.reshape(1024, 1)); const("w_conv", w_conv); const("w_move", w_move)
const("w_att2", w_att2); const("w_guard", w_guard); const("w_pick", w_pick); const("edge", edge)
const("ones3", np.ones((1, 1, 3, 3), np.float32)); const("w_pol", w_pol.reshape(1024, 64))
const("zeros", np.zeros((1, 64, 64), np.float32))
const("flat_shape", np.array([-1, 1024], np.int64)); const("pol_shape", np.array([-1, 4096], np.int64))
const("val_shape", np.array([-1], np.int64)); const("ax1", np.array([1], np.int64))
const("axes", np.array([1, 2, 3], np.int64))
for n, v in dict(z=0, o=1, two=2, half=0.5, gain=P["gain"], loss=P["loss"], ku=P["ku"], kg=P["kg"],
                 kb=P["kb"], fw=P["fw"], inv=1 / P["scale"]).items():
    const(n, f(v))

node("Reshape", ["observation", "flat_shape"], "flat")
node("Reshape", [node("MatMul", ["flat", "w_lin"], "lin2"), "val_shape"], "lin")
node("Conv", ["observation", "w_conv"], "maps", pads=[3, 3, 3, 3])
node("Split", ["maps"], ["oa_r", "ea_r", "ov", "ev"], axis=1, num_outputs=4)
node("Clip", ["oa_r", "z", "o"], "oa"); node("Clip", ["ea_r", "z", "o"], "ea")
node("ReduceMax", [node("Mul", ["oa", "ev"], "he"), "axes"], "gain_v", keepdims=0)
node("ReduceMax", [node("Mul", ["ea", "ov"], "ho"), "axes"], "loss_v", keepdims=0)
# 2-action reach
node("Clip", [node("Conv", ["observation", "w_move"], "mv_r", pads=[3, 3, 3, 3]), "z", "o"], "mv")
node("Clip", [node("Conv", ["mv", "w_att2"], "r2_r", pads=[3, 3, 3, 3]), "z", "o"], "r2")
node("Split", ["r2"], ["reach_own", "reach_en"], axis=1, num_outputs=2)
node("Conv", ["observation", "w_guard"], "gd", pads=[1, 1, 1, 1])
node("Split", ["gd"], ["gd_own", "gd_en"], axis=1, num_outputs=2)
node("Conv", ["observation", "w_pick"], "pk")
node("Split", ["pk"], ["own_k", "en_k", "locked", "actl", "urgp", "side"], axis=1, num_outputs=6)
node("ReduceMax", [node("Mul", ["own_k", "reach_en"], "okr_m"), "axes"], "okr", keepdims=0)
node("ReduceMax", [node("Mul", ["en_k", "reach_own"], "ekr_m"), "axes"], "ekr", keepdims=0)
node("ReduceSum", [node("Mul", ["own_k", "gd_own"], "og_m"), "axes"], "og", keepdims=0)
node("ReduceSum", [node("Mul", ["en_k", "gd_en"], "eg_m"), "axes"], "eg", keepdims=0)
node("Sub", ["o", node("Clip", ["og", "z", "o"], "og1")], "own_ung")
node("Sub", ["o", node("Clip", ["eg", "z", "o"], "eg1")], "en_ung")
node("Sub", [node("Mul", ["two", node("ReduceMean", ["actl", "axes"], "a", keepdims=0)], "a2x"), "o"], "a2")
node("ReduceMean", ["urgp", "axes"], "urg", keepdims=0)
node("ReduceMean", ["side", "axes"], "m", keepdims=0)
# own kernel danger: -okr * (ku*ung + kg*(1-ung)) * (1 - 0.5*a2)
node("Add", [node("Mul", ["own_ung", "ku"], "t1"), node("Mul", [node("Sub", ["o", "own_ung"], "own_g"), "kg"], "t2")], "kd")
node("Sub", ["o", node("Mul", ["a2", "half"], "ha2")], "esc")
node("Mul", [node("Mul", ["okr", "kd"], "t3"), "esc"], "own_danger")
node("Mul", [node("Mul", [node("Mul", ["ekr", "en_ung"], "t4"), "a2"], "t5"), "kb"], "en_danger")
# firewall: next ring = unlocked & (edge | next to locked)
node("Clip", [node("Add", ["edge", node("Conv", ["locked", "ones3"], "nl", pads=[1, 1, 1, 1])], "nr_r"), "z", "o"], "nr1")
node("Mul", ["nr1", node("Sub", ["o", "locked"], "unl")], "nring")
node("ReduceMax", [node("Mul", ["own_k", "nring"], "of_m"), "axes"], "ofire", keepdims=0)
node("ReduceMax", [node("Mul", ["en_k", "nring"], "ef_m"), "axes"], "efire", keepdims=0)
node("Mul", ["ofire", "efire"], "both")
node("Sub", ["o", "ofire"], "nof"); node("Sub", ["o", "efire"], "nef")
node("Sub", [node("Mul", ["efire", "nof"], "fw1"), node("Mul", ["ofire", "nef"], "fw2")], "fw3")
node("Add", ["fw3", node("Mul", ["both", node("Sub", [node("Mul", ["two", "m"], "m2"), "o"], "msign")], "fw4")], "fw5")
node("Mul", ["urg", "urg"], "u2"); node("Mul", ["u2", "u2"], "u4")
node("Mul", [node("Mul", ["fw5", "u4"], "fw6"), "fw"], "fire")
# total
node("Add", ["lin", node("Mul", ["gain_v", "gain"], "g")], "v1")
node("Sub", ["v1", node("Mul", ["loss_v", "loss"], "l")], "v2")
node("Sub", ["v2", "own_danger"], "v3")
node("Add", ["v3", "en_danger"], "v4")
node("Add", ["v4", "fire"], "v5")
node("Tanh", [node("Mul", ["v5", "inv"], "v6")], "value")
node("Unsqueeze", [node("MatMul", ["flat", "w_pol"], "tb"), "ax1"], "tb3")
node("Reshape", [node("Add", ["tb3", "zeros"], "pol3"), "pol_shape"], "policy")

graph = helper.make_graph(
    nodes, "colony_v5",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out)
print("saved", out, P)

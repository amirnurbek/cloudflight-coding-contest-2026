"""Train a small residual CNN on self-play data as a correction on top of the v5 heuristic.

value  = tanh(heuristic_raw + net_raw)       policy = net_policy (+ alpha * heuristic policy)
Exports one merged ONNX (heuristic graph + net nodes) built with onnx.helper (allow-listed ops only).

    python train/train.py base.onnx out.onnx [epochs]
"""
import glob
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper, numpy_helper

BASE, OUT = sys.argv[1], sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
import os
C, BLOCKS, QK = int(os.environ.get('C', 48)), int(os.environ.get('BLOCKS', 4)), 8
torch.manual_seed(0)

# ---------- data ----------
files = sorted(glob.glob("train/data*/*.npz"))
obs, pidx, pp, z = [], [], [], []
for f in files:
    d = np.load(f)
    obs.append(d["obs"]); pidx.append(d["pidx"]); pp.append(d["pp"]); z.append(d["z"])
obs = np.concatenate(obs).astype(np.float32) / 8.0
pidx = np.concatenate(pidx).astype(np.int64); pp = np.concatenate(pp).astype(np.float32)
z = np.concatenate(z).astype(np.float32)
print(f"{len(files)} files, {len(z)} positions", flush=True)

# heuristic pre-tanh value for every position (base graph exposes value = tanh(v6))
sess = ort.InferenceSession(BASE, providers=["CPUExecutionProvider"])
hv = np.concatenate([sess.run(["value"], {"observation": obs[i:i + 8192]})[0] for i in range(0, len(obs), 8192)])
h = np.arctanh(np.clip(hv, -0.995, 0.995)).astype(np.float32)

# transpose augmentation (all piece patterns and rings are symmetric under (r,c)->(c,r))
sq = np.arange(64); tsq = (sq % 8) * 8 + sq // 8
def t_idx(ix):
    valid = ix >= 0
    f_, t_ = ix // 64, ix % 64
    return np.where(valid, tsq[f_] * 64 + tsq[t_], -1)
obs = np.concatenate([obs, obs.transpose(0, 1, 3, 2)])
pidx = np.concatenate([pidx, t_idx(pidx)]); pp = np.concatenate([pp, pp]); z = np.concatenate([z, z]); h = np.concatenate([h, h])

n = len(z); perm = np.random.default_rng(0).permutation(n)
val_n = min(20000, n // 10)
vi, ti = perm[:val_n], perm[val_n:]


# ---------- net ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(16, C, 3, padding=1)
        self.blocks = nn.ModuleList([nn.ModuleList([nn.Conv2d(C, C, 3, padding=1), nn.Conv2d(C, C, 3, padding=1)]) for _ in range(BLOCKS)])
        self.v1 = nn.Conv2d(C, 8, 1); self.v2 = nn.Linear(512, 64); self.v3 = nn.Linear(64, 1)
        self.q = nn.Conv2d(C, QK, 1); self.k = nn.Conv2d(C, QK, 1)
        nn.init.zeros_(self.v3.weight); nn.init.zeros_(self.v3.bias)

    def forward(self, x):
        y = F.relu(self.stem(x))
        for a, b in self.blocks:
            y = F.relu(y + 0.5 * b(F.relu(a(y))))
        v = self.v3(F.relu(self.v2(torch.flatten(F.relu(self.v1(y)), 1)))).squeeze(1)
        q = self.q(y).flatten(2).transpose(1, 2)           # (B,64,QK)
        k = self.k(y).flatten(2)                           # (B,QK,64)
        pol = torch.bmm(q, k).flatten(1)                   # (B,4096) = from*64+to
        return pol, v


dev = "mps" if torch.backends.mps.is_available() else "cpu"
net = Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
T = lambda a: torch.from_numpy(a).to(dev)
steps = EPOCHS * (len(ti) // 512)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)


def batch_loss(ix):
    x, pi, pv, zz, hh = T(obs[ix]), T(pidx[ix]), T(pp[ix]), T(z[ix]), T(h[ix])
    pol, v = net(x)
    value = torch.tanh(hh + v)
    lv = F.mse_loss(value, zz)
    logp = F.log_softmax(pol, 1)
    mask = pi >= 0
    lp = -(torch.gather(logp, 1, pi.clamp(min=0)) * pv * mask).sum(1).mean()
    base = F.mse_loss(torch.tanh(hh), zz)
    return lv, lp, base


t0 = time.time()
for ep in range(EPOCHS):
    net.train(); rng = np.random.default_rng(ep)
    sh = rng.permutation(ti)
    for s in range(0, len(sh) - 511, 512):
        lv, lp, _ = batch_loss(sh[s:s + 512])
        opt.zero_grad(); (lv * 2 + lp).backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        r = [batch_loss(vi[s:s + 2048]) for s in range(0, len(vi), 2048)]
    lv, lp, base = (float(np.mean([float(x[i]) for x in r])) for i in range(3))
    print(f"epoch {ep}: val value-mse {lv:.4f} (heuristic alone {base:.4f}), policy-ce {lp:.3f} [{time.time()-t0:.0f}s]", flush=True)

# ---------- export merged ONNX ----------
net = net.cpu().eval()
m = onnx.load(BASE); g = m.graph
# detach heuristic outputs: rename final value/policy producers
for nd in g.node:
    nd.output[:] = ["h_value" if o == "value" else "h_policy" if o == "policy" else o for o in nd.output]
W = lambda t: t.detach().numpy().astype(np.float32)
inits = []
def const(name, a):
    g.initializer.append(numpy_helper.from_array(np.asarray(a), name)); return name
def node(op, ins, out, **kw):
    g.node.append(helper.make_node(op, ins, [out], **kw)); return out
def conv(x, layer, name, pad):
    const(name + "_w", W(layer.weight)); const(name + "_b", W(layer.bias))
    return node("Conv", [x, name + "_w", name + "_b"], name, pads=[pad] * 4)

y = node("Relu", [conv("observation", net.stem, "n_stem", 1)], "n_y0")
const("n_half", np.float32(0.5))
for i, (a, b) in enumerate(net.blocks):
    t = node("Relu", [conv(y, a, f"n_b{i}a", 1)], f"n_b{i}ar")
    t = conv(t, b, f"n_b{i}b", 1)
    y = node("Relu", [node("Add", [y, node("Mul", [t, "n_half"], f"n_b{i}h")], f"n_b{i}s")], f"n_y{i+1}")
v = node("Flatten", [node("Relu", [conv(y, net.v1, "n_v1", 0)], "n_v1r")], "n_vf", axis=1)
const("n_v2w", W(net.v2.weight)); const("n_v2b", W(net.v2.bias))
v = node("Relu", [node("Gemm", [v, "n_v2w", "n_v2b"], "n_v2", transB=1)], "n_v2r")
const("n_v3w", W(net.v3.weight)); const("n_v3b", W(net.v3.bias))
v = node("Gemm", [v, "n_v3w", "n_v3b"], "n_v3", transB=1)
const("n_vs", np.array([-1], np.int64))
v = node("Reshape", [v, "n_vs"], "n_raw")
node("Tanh", [node("Add", ["v6", v], "n_sum")], "value")
const("n_qs", np.array([-1, QK, 64], np.int64))
q = node("Transpose", [node("Reshape", [conv(y, net.q, "n_q", 0), "n_qs"], "n_q3")], "n_qt", perm=[0, 2, 1])
k = node("Reshape", [conv(y, net.k, "n_k", 0), "n_qs"], "n_k3")
const("n_ps", np.array([-1, 4096], np.int64))
pol = node("Reshape", [node("MatMul", [q, k], "n_qk"), "n_ps"], "n_pol")
const("n_alpha", np.float32(ALPHA))
node("Add", [pol, node("Mul", ["h_policy", "n_alpha"], "n_hp")], "policy")
onnx.checker.check_model(m)
onnx.save(m, OUT)
print("saved", OUT, flush=True)

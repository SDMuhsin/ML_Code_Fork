---
id: 2m
title: Training with the accelerated step
minutes: 28
---
# Intro
This module puts the pieces of `2j`-`2l` together on a flow with a known
answer, trains it with your own jet and your own emitted adjoint, and then
measures how much time and memory each part saved, against nested autograd.

# Shared
```python
import logging
import torch.nn as nn
logging.getLogger("torch.utils.flop_counter").disabled = True   # silences a "triton not found" notice
from torch.utils.flop_counter import FlopCounterMode

def demo_net(din=1, width=32, depth=3, dout=1):
    # demo_net from 2d, with dout outputs instead of 1.
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)

def ddx(u, x):
    # Your ddx from 2d-3.
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

def run_net(net, x):
    # net(x) with plain function calls, as in 2g (FlopCounterMode needs this).
    for layer in net:
        x = F.linear(x, layer.weight, layer.bias) if isinstance(layer, nn.Linear) else torch.tanh(x)
    return x

def count_saved(fn):
    # From 2f: (number of saves, distinct bytes saved) while fn() runs.
    store, n = {}, 0
    def pack(t):
        nonlocal n
        n += 1
        s = t.untyped_storage()
        store[s.data_ptr()] = s.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return n, sum(store.values())

def flops(fn):
    # From 2g: matrix-product FLOPs counted while fn() runs, and fn's result.
    with FlopCounterMode(display=False) as fc:
        out = fn()
    return fc.get_total_flops(), out

NS_NAMES = ["u", "v", "p", "u_x", "u_y", "v_x", "v_y", "p_x", "p_y", "u_xx", "u_yy", "v_xx", "v_yy"]

def make_ns_loss(nu):
    # From 2l: per-point loss of 2D steady incompressible Navier-Stokes.
    nu = float(nu)
    def loss_point(Fd):
        u, v = Fd["u"], Fd["v"]
        r1 = u * Fd["u_x"] + v * Fd["u_y"] + Fd["p_x"] - nu * (Fd["u_xx"] + Fd["u_yy"])
        r2 = u * Fd["v_x"] + v * Fd["v_y"] + Fd["p_y"] - nu * (Fd["v_xx"] + Fd["v_yy"])
        r3 = Fd["u_x"] + Fd["v_y"]
        return r1 * r1 + r2 * r2 + r3 * r3
    return loss_point

def as_dict(stack, names):
    # (N, n) field stack -> {name: (N, 1) column}
    return {n: stack[:, i:i + 1] for i, n in enumerate(names)}

def forward_line(entry):
    # From 2l-3 Shared.
    op, out = entry[0], entry[-1]
    if op == "add":
        return f"{out.name} = {entry[1].name} + {entry[2].name}"
    if op == "sub":
        return f"{out.name} = {entry[1].name} - {entry[2].name}"
    if op == "mul":
        return f"{out.name} = {entry[1].name} * {entry[2].name}"
    if op == "smul":
        return f"{out.name} = {entry[1]!r} * {entry[2].name}"
    if op == "cadd":
        return f"{out.name} = {entry[1].name} + {entry[2]!r}"
    raise ValueError(f"unknown op {op}")

def ns_stack(J):
    # (5, N, 3) jet of (u, v, p) -> (N, 13) field stack in NS_NAMES order.
    return torch.cat([J[0], J[1, :, 0:1], J[2, :, 0:1], J[1, :, 1:2], J[2, :, 1:2], J[1, :, 2:3], J[2, :, 2:3],
                      J[3, :, 0:1], J[4, :, 0:1], J[3, :, 1:2], J[4, :, 1:2]], 1)

def nested_stack(net, xy):
    # The same (N, 13) stack by nested autograd (2i), for comparisons.
    x = xy.detach().requires_grad_()
    out = run_net(net, x)
    cols = {}
    for k, f in enumerate("uvp"):
        u = out[:, k:k + 1]
        g = ddx(u, x)
        cols[f], cols[f + "_x"], cols[f + "_y"] = u, g[:, 0:1], g[:, 1:2]
        if f != "p":
            cols[f + "_xx"] = ddx(g[:, 0:1], x)[:, 0:1]
            cols[f + "_yy"] = ddx(g[:, 1:2], x)[:, 1:2]
    return torch.cat([cols[n] for n in NS_NAMES], 1)

RE = 40.0
LAM = RE / 2 - math.sqrt(RE ** 2 / 4 + 4 * math.pi ** 2)

def exact(xy):
    # Kovasznay flow at Re = 40: (N, 2) -> (N, 3) columns u, v, p.
    x, y = xy[:, 0:1], xy[:, 1:2]
    e = torch.exp(LAM * x)
    return torch.cat([1 - e * torch.cos(2 * math.pi * y), LAM / (2 * math.pi) * e * torch.sin(2 * math.pi * y),
                      0.5 * (1 - torch.exp(2 * LAM * x))], 1)

def weight_grads(net):
    # All weight gradients of net as one flat vector.
    return torch.cat([p.grad.flatten() for p in net.parameters()])

def rel_err(a, b):
    # |a - b| / |b| over all entries, as in 2d.
    return ((a - b).norm() / b.norm()).item()
```

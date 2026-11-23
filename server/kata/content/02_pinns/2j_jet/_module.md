---
id: 2j
title: Derivatives in the forward pass
minutes: 36
---
# Intro
In `2g` a second derivative cost a pass over the graph of a pass. `2j-3`
shows that a network with several outputs pays this once per output. This
module computes `u`,
`u_x` and `u_xx` in the forward pass itself, with no `autograd.grad` call, by
carrying the derivatives along with the values from layer to layer. By the
end you can write that forward for any tanh MLP, check it against `ddx`, and
say what it costs.

This module and the next three build the training step of the
`pinn-solver-accel` code (the SAGE pull request, branch `usask/accel-pr`,
package `usask_accel/`), called here the **accelerated step**. It has three
parts. A Taylor jet (`2j`) removes the graph of a graph. **SAGE** (`2l`)
replaces the autograd graph of the residual arithmetic with generated code.
One ordinary backward pass (`2k`) joins the two.

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

def nested_2d(net, xy):
    # Nested autograd, as in 2i: for each output k, u, u_x, u_y, u_xx, u_yy as a list of (N, 1) tensors.
    x = xy.detach().requires_grad_()
    out = run_net(net, x)
    res = []
    for k in range(out.shape[1]):
        u = out[:, k:k + 1]
        g = ddx(u, x)
        res.append([u, g[:, 0:1], g[:, 1:2], ddx(g[:, 0:1], x)[:, 0:1], ddx(g[:, 1:2], x)[:, 1:2]])
    return res
```

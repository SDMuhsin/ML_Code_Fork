---
id: 2i
title: The heat equation
minutes: 28
---
# Intro
The PINNs so far had one input. Most physics has several: position and
time, or two or three space coordinates. This module trains a PINN with two
inputs, `x` and `t`, for the heat equation, and ends with what its training
step costs, in the units of `2g`, and when a PINN is worth using instead of
a classic solver.

# Shared
```python
import logging
import torch.nn as nn
logging.getLogger("torch.utils.flop_counter").disabled = True   # silences a "triton not found" notice
from torch.utils.flop_counter import FlopCounterMode

def demo_net(din=1, width=32, depth=3):
    # The same layers as your MLP() from 2b-2. Demos cannot see your code, so they use this.
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)

def ddx(u, x):
    # Your ddx from 2d-3 (demos cannot see your code, so Shared defines it too).
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

def run_net(net, x):
    # net(x) written with plain function calls, as in 2g: FlopCounterMode cannot
    # count torch.autograd.grad through nn.Module calls.
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
```

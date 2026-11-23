---
id: 2k
title: The split step
minutes: 20
---
# Intro
After `2j` the step has no graph of a graph, but autograd still records a
graph for the residual arithmetic: every product saves its operands. In this
lesson's Navier-Stokes step that graph is about 1% of the saved memory and a
small share of the time (`2m-2` measures both). In the real code the
residual has 171 operations, and the accelerated step replaces its graph
with generated straight-line code. A compiler (`jax.jit`, a program that
rewrites the whole step before it runs; `2m-2`) can merge that code into a
few GPU programs. To do that, the step must be split where the
residual starts.

The jet gives the fields a residual reads: `u`, `u_x`, `u_xx`, ... as
`(N, 1)` tensors. The loss is arithmetic on them. This module splits the
weight backward in two: first the derivative of the loss with respect to
each field at every point, then one ordinary backward pass through the jet
that turns those into weight gradients. By the end you can write the first
half by hand for a small residual and check the step against
`loss.backward()`.

# Shared
```python
import torch.nn as nn

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

def weight_grads(net):
    # All weight gradients of net as one flat vector.
    return torch.cat([p.grad.flatten() for p in net.parameters()])

def rel_err(a, b):
    # |a - b| / |b| over all entries, as in 2d.
    return ((a - b).norm() / b.norm()).item()
```

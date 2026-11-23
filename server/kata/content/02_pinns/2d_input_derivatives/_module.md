---
id: 2d
title: Derivatives with respect to inputs
minutes: 37
---
# Intro
A physics law talks about how a quantity changes in space or time: `du/dx`,
the derivative of the network's output with respect to its input. By the end
you can compute it with `torch.autograd.grad`, say why one call gives the
derivative at every point at once, and put it in a loss that trains.

Two names are fixed from here on. The **weight gradient** is the derivative
of the loss with respect to a parameter (what `2a`-`2c` computed). The
**input derivative** is the derivative of the network's output `u` with
respect to its input `x`.

# Shared
```python
import torch.nn as nn

def demo_net(din=1, width=32, depth=3):
    # The same layers as your MLP() from 2b-2. Demos cannot see your code, so they use this.
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)

def fd_grad(loss_fn, W, h=1e-6):
    # Finite-difference derivative of loss_fn() with respect to every entry of W (central differences).
    g = torch.zeros_like(W)
    flat, gflat = W.data.view(-1), g.view(-1)
    for i in range(flat.numel()):
        old = flat[i].item()
        flat[i] = old + h; lp = loss_fn().item()
        flat[i] = old - h; lm = loss_fn().item()
        flat[i] = old
        gflat[i] = (lp - lm) / (2 * h)
    return g

def rel_err(a, b):
    # Relative size of the difference: |a - b| / |b|, over all entries.
    return ((a - b).norm() / b.norm()).item()
```

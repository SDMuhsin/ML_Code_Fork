---
id: 2f
title: What autograd saves
minutes: 18
---
# Intro
Two names for the rest of the lesson. A **derivative pass** is a backward
pass toward the input, such as the `ddx` call that gives `u_x`. The **weight
backward** is the final `loss.backward()` that fills the weight gradients. A
Poisson step in `2e-3` records two derivative passes before the weight
backward; a plain fit step records none. This module and the next measure
what that costs, memory first. For networks of the size in this lesson,
training memory is mostly the saved values of `2d-3`: forward values autograd
keeps for the backward pass. (For a large model with a small batch, the
parameters, their gradients and Adam's state, about 4 times the parameter
memory, can be the larger part.) By the end you can say which tensors each operation saves and why, and
predict an MLP's saved memory from its size before running it.

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

def count_saved(fn):
    # Runs fn() and records every tensor autograd saves for the backward pass.
    # Returns (number of saves, bytes of the distinct memory those tensors use).
    store, n = {}, 0
    def pack(t):
        nonlocal n
        n += 1
        s = t.untyped_storage()          # the memory block behind the tensor
        store[s.data_ptr()] = s.nbytes()  # keyed by address: a block saved twice counts once
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return n, sum(store.values())
```

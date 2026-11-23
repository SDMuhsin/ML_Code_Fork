---
id: 2e
title: A first PINN
minutes: 45
---
# Intro
A **physics-informed neural network (PINN)** is an MLP trained so that a
physics equation holds at chosen points, instead of trained to match
measured data. By the end you have trained two: one for an equation in
time and one for an equation in space, each checked against its exact
solution.

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
```

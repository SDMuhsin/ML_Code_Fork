---
id: 2h
title: When a PINN fails
minutes: 23
---
# Intro
A PINN whose loss goes down can still be wrong, and the total loss
does not say why. This module shows two failures on the Poisson equation of
`2e-3` and how to read them from the printed parts of the loss: the residual
loss, the boundary loss, and the max error against the exact solution.

# Shared
```python
import torch.nn as nn

def demo_net(din=1, width=32, depth=3, act=nn.Tanh):
    # The same layers as your MLP() from 2b-2. Demos cannot see your code, so they use this.
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)

def ddx(u, x):
    # Your ddx from 2d-3 (demos cannot see your code, so Shared defines it too).
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

def train_poisson(net, w_bc=1.0, u_end=0.0, steps=500, lr=1e-2, show=(0, 250, 499), hard=False):
    # Trains u'' = -pi^2 sin(pi x) on [0, 1] with u(0) = u(1) = u_end, 64 points, Adam.
    # Loss = residual loss + w_bc * boundary loss. Returns the last (residual loss, boundary loss, max error).
    # hard=True trains u(x) = u_end + x (1 - x) net(x) instead, which meets both boundary values for any weights.
    model = (lambda z: u_end + z * (1 - z) * net(z)) if hard else net
    x = torch.linspace(0, 1, 64)[:, None].requires_grad_()
    f = -math.pi ** 2 * torch.sin(math.pi * x.detach())
    xb = torch.tensor([[0.0], [1.0]])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        res_loss = ((ddx(ddx(model(x), x), x) - f) ** 2).mean()
        bc_loss = ((model(xb) - u_end) ** 2).mean()
        (res_loss + w_bc * bc_loss).backward()
        opt.step()
        if step in show:
            print(f"step {step:3d}: residual loss {res_loss.item():9.4g}, boundary loss {bc_loss.item():9.3g}")
    x_test = torch.linspace(0, 1, 201)[:, None]
    with torch.no_grad():
        err = (model(x_test) - u_end - torch.sin(math.pi * x_test)).abs().max().item()
    return res_loss.item(), bc_loss.item(), err
```

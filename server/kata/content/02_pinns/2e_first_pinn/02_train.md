---
title: Train it
difficulty: Medium
devices: [cpu]
uses: [2b-2, 2d-3, 2e-1]
timeout: 60
inspect: [res_losses, ic_losses, max_err]
---
# Lesson
## What should the loss be?

The PINN loss has one term per condition:

    L = mean(r(t)**2) + (u(0) - 1)**2

The first term, the **residual loss**, is the mean squared residual over the
collocation points. The second, the **initial loss**, is the squared miss of
the initial condition. Neither is enough alone. The demo evaluates both on
three functions:

```python demo
t = torch.linspace(0, 2, 64)[:, None].requires_grad_()
t0 = torch.zeros(1, 1)
def parts(u_fn):
    u = u_fn(t)
    r = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] + 2 * u
    return round((r ** 2).mean().item(), 6), round(((u_fn(t0) - 1) ** 2).sum().item(), 6)
print("u = 0            residual loss, initial loss:", parts(lambda s: 0 * s))
print("u = 1.5 exp(-2t) residual loss, initial loss:", parts(lambda s: 1.5 * torch.exp(-2 * s)))
print("u = exp(-2t)     residual loss, initial loss:", parts(lambda s: torch.exp(-2 * s)))
```

`u = 0` and `1.5 exp(-2t)` both solve the ODE, so their residual loss is 0;
only the initial loss tells them apart. Only `exp(-2t)` makes both 0.
The two terms are added with weight 1. That works here; module `2h` shows a
case where it does not, and how to weight them.

The training loop is the `2c` loop with this loss in place of the mean
squared error. No target values appear anywhere: the equation and the
initial condition are the only information the network gets. Each step
evaluates the network at the 64 collocation points and at `t = 0`, takes the
input derivative with your `ddx` (which keeps its graph), and calls
`backward()` once on the sum of the two parts. This problem uses lr 5e-3.

Record the two parts separately. When a run goes wrong, the part that stays
large tells you which condition the network failed.

The collocation points carry no measured data, so you can pick any points,
and you can pick new ones every step. With a fixed grid the network can
drive the residual to 0 at those 64 points and still be wrong between them.
Drawing fresh points each step (`t = 2 * torch.rand(64, 1)`, then
`requires_grad_()`) checks the equation over the whole interval during the
run, at the same cost per step. The residual loss then varies from step to
step with the points drawn, so judge such a run by the error on a fixed test
grid, not by the last loss value. This problem keeps the fixed grid so that
runs are repeatable.

# Task
Train the given `net` for 500 steps with Adam, lr 5e-3, on the loss above,
using your `residual` (it reads `K` from Given). Each step, append the two
parts as Python floats to `res_losses` and `ic_losses`. Afterwards set
`max_err`: the largest `|net(t) - exp(-2t)|` on the 201 points `t_test`
(compute it under `torch.no_grad()`).

# Given
```python
torch.set_num_threads(1)
K = 2.0
t = torch.linspace(0, 2, 64)[:, None].requires_grad_()   # collocation points
t0 = torch.zeros(1, 1)                                   # where the initial condition holds
t_test = torch.linspace(0, 2, 201)[:, None]
torch.manual_seed(0)
net = MLP()
```

# Starter
```python
opt = torch.optim.Adam(net.parameters(), lr=5e-3)
res_losses, ic_losses = [], []
for step in range(500):
    opt.zero_grad()
    res_loss = TODO
    ic_loss = TODO
    # TODO: total loss, backward, step; record both parts
max_err = TODO
```

# Solution
```python
opt = torch.optim.Adam(net.parameters(), lr=5e-3)
res_losses, ic_losses = [], []
for step in range(500):
    opt.zero_grad()
    res_loss = (residual(net, t) ** 2).mean()
    ic_loss = ((net(t0) - 1) ** 2).sum()
    loss = res_loss + ic_loss
    loss.backward()
    opt.step()
    res_losses.append(res_loss.item())
    ic_losses.append(ic_loss.item())
with torch.no_grad():
    max_err = (net(t_test) - torch.exp(-2 * t_test)).abs().max().item()
print(f"step 499: residual loss {res_losses[-1]:.2e}, initial loss {ic_losses[-1]:.2e}, max error {max_err:.2e}")
```

# Checks
```python
with torch.no_grad():
    u_test = net(t_test)
    true_err = (u_test - torch.exp(-2 * t_test)).abs().max().item()
    u_start, u_end = net(t0).item(), net(t_test[-1:]).item()
expect("one entry per step in both lists", len(res_losses) == len(ic_losses) and 0 < len(res_losses) <= 1000, f"{len(res_losses)} residual and {len(ic_losses)} initial entries; want one per step, at most 1000 steps")
expect("u(0) is about 1", abs(u_start - 1) < 0.05, f"u(0) = {u_start:.3f}, want about 1. " + ("Near 0: the network found u = 0, which satisfies the ODE. Is the initial term in the loss you call backward() on?" if abs(u_start) < 0.3 else (f"u(2) = {u_end:.3f}: u does not decay. Check the sign of the residual, du/dt + K u." if u_end > 0.1 else "Look at ic_losses: did the initial loss go down?")))
expect("u(2) is about exp(-4) = 0.018", abs(u_end - math.exp(-4)) < 0.03, f"u(2) = {u_end:.3f}. " + ("u does not decay: check the sign of the residual, du/dt + K u." if u_end > 0.1 else "Compare the last entries of res_losses and ic_losses: the larger one names the condition the network missed."))
expect("max error < 0.05", true_err < 0.05, f"max |net - exp(-2t)| = {true_err:.3g}; last residual loss {res_losses[-1]:.2g}, last initial loss {ic_losses[-1]:.2g}. The larger part names the condition the network missed.")
check("max_err", max_err, want=true_err, atol=1e-6)
```

# Hints
1. The residual loss is `(residual(net, t)**2).mean()`; the initial loss uses `net(t0)`.
2. Call `backward()` on the sum of the two parts; record each part with `.item()`.

# Takeaway
Over seeds 0-9 the max error was 0.0006-0.010, about 2 s per run. Two
mistakes and what they gave: no initial term, error 1.0 (the network found
`u = 0`); the residual written `u' - 2u`, error 0.67-1.4.

Each step cost more than a plain fit. Measured on this machine, CPU, 64
points, `MLP()`: a plain sine fit step 1.8 ms, this step 3.2 ms, and a step
with a second derivative (next problem) 4.4 ms. Modules `2f` and `2g` take
the extra cost apart.

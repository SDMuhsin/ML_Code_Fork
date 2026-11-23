---
title: The residual of an ODE
difficulty: Easy
uses: [2b-2, 2d-3]
inspect: [residual]
---
# Lesson
## How do you measure "the equation holds"?

An **ODE** (ordinary differential equation) is an equation that relates a
function of one variable to its derivatives. The first PINN solves

    u'(t) = -k u(t),   k = 2,   t in [0, 2]

In words: the rate of change of `u` is proportional to `u` and has the
opposite sign, so `u` decays toward 0, faster when it is large. Every
`u = C exp(-2t)` satisfies it. The **initial condition** `u(0) = 1` picks one
of them: `u = exp(-2t)`.

A network will not satisfy the ODE exactly. To measure how far off it is,
move everything to one side. The **residual** is

    r(t) = u'(t) + k u(t)

It is 0 at every `t` exactly when `u` solves the ODE. The points where you
evaluate it are the **collocation points**; here 64 evenly spaced points in
`[0, 2]`, shape `(64, 1)`.

```python demo
k = 2.0
t = torch.linspace(0, 2, 64)[:, None].requires_grad_()
def r(u):
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    return u_t + k * u
print("exact exp(-2t): max |r| =", r(torch.exp(-k * t)).abs().max().item())
r_wrong = r(torch.exp(-t))
print("exp(-t):        max |r - exp(-t)| =", (r_wrong - torch.exp(-t)).abs().max().item())
```

The exact solution has residual 0 (here exactly; in general up to float32
rounding). `exp(-t)` decays at the wrong rate: its residual is `-exp(-t) + 2 exp(-t) =
exp(-t)`, as printed. Before training anything, check the residual on a
function whose answer you know. A wrong sign or a wrong shape in the
residual is otherwise blamed on the training.

# Task
`residual(u_fn, t)`: `u_fn` maps a `(N, 1)` tensor to a `(N, 1)` tensor (a
network or a formula); `t` is `(N, 1)` and already has `requires_grad`.
Return `u'(t) + K * u(t)` with shape `(N, 1)`, using your `ddx`. `K` is in
Given.

# Given
```python
K = 2.0
```

# Starter
```python
def residual(u_fn, t):
    TODO
```

# Solution
```python
def residual(u_fn, t):
    u = u_fn(t)
    return ddx(u, t) + K * u
```

# Checks
```python
ts = torch.linspace(0, 2, 64)[:, None].requires_grad_()
td = ts.detach()
check("residual of exp(-2t) is about 0", residual(lambda s: torch.exp(-2 * s), ts), want=torch.zeros(64, 1), shape=(64, 1), atol=1e-5)
check("residual of exp(-t) is exp(-t)", residual(lambda s: torch.exp(-s), ts), want=torch.exp(-td), atol=1e-5)
torch.manual_seed(0)
rnet = MLP()
r_net = residual(rnet, ts)
same("residual of an MLP has shape (N, 1)", S(r_net), (64, 1))
check("residual of an MLP", r_net, want=ddx(rnet(ts), ts) + 2 * rnet(ts), atol=1e-5)
expect("the residual can be trained (it has a graph)", r_net.grad_fn is not None, "the result has no grad_fn: was the derivative or u detached?")
```

# Hints
1. Two lines: compute `u` from `t`, then combine `ddx(u, t)` and `u`.
2. If the `exp(-t)` check gives `-3 exp(-t)`, the sign of one term is flipped.

# Takeaway
`residual` returns one number per collocation point, shape `(64, 1)`. It is 0
for the exact solution and equal to `exp(-t)` for the wrong one. With
`u'(t) - K u(t)` (wrong
sign) the exact solution would have residual `-4 exp(-2t)`, which the first
check catches.

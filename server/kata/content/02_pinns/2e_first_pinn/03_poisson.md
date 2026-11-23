---
title: Poisson in 1D
difficulty: Medium
devices: [cpu]
uses: [2b-2, 2d-3]
timeout: 60
inspect: [d2dx2, losses, max_err]
---
# Lesson
## An equation with a second derivative

The **Poisson equation** in one dimension gives the second derivative
(the curvature) of `u` everywhere:

    u''(x) = -pi**2 sin(pi x),   x in [0, 1],   u(0) = u(1) = 0

Physically, `u(x)` is the steady temperature along a rod whose two ends are
held at 0. Heat is fed in along the rod, most in the middle. In steady state
the heat fed in at a point equals `-u''` there, so the temperature bulges
upward. The right-hand side, `f(x) = -pi**2 sin(pi x)`, is called the
**source term**.

The two end values are **boundary conditions**: values `u` must take at the
edges of the interval. The exact solution is `u = sin(pi x)`. With one variable this is an ODE; in 2D or 3D
`u''` becomes `u_xx + u_yy (+ u_zz)` and it is a PDE.

A second derivative is `ddx` applied to the result of `ddx`. Both calls need
`create_graph=True`: the outer call differentiates the inner result, and the
loss differentiates the outer result. Your `ddx` from `2d-3` already does
this. The demo writes the calls out:

```python demo
x = torch.linspace(0, 1, 5)[:, None].requires_grad_()
ddx = lambda u, x: torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
u = torch.sin(3 * x)
print("max |u'' - (-9 sin(3x))| =", (ddx(ddx(u, x), x) + 9 * torch.sin(3 * x)).abs().max().item())
```

## Why the boundary term is needed

Add `a x + b` to the exact solution. Its second derivative is 0, so the
residual does not change. The residual alone allows a whole family of
solutions; the boundary conditions pick `a = b = 0`. The demo checks one
member of the family, then trains for 500 steps with the boundary term left
out (about 2 seconds):

```python demo
torch.set_num_threads(1)
xs = torch.linspace(0, 1, 64)[:, None].requires_grad_()
src = -math.pi ** 2 * torch.sin(math.pi * xs.detach())
shifted = torch.sin(math.pi * xs) + 0.3 * xs + 0.2
print("residual of sin(pi x) + 0.3x + 0.2: max", (ddx(ddx(shifted, xs), xs) - src).abs().max().item())
torch.manual_seed(0)
net = demo_net()
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for step in range(500):
    opt.zero_grad()
    u = net(xs)
    res_loss = ((ddx(ddx(u, xs), xs) - src) ** 2).mean()
    res_loss.backward()
    opt.step()
    if step == 0:
        print(f"step 0:   residual loss {res_loss.item():.1f}")
with torch.no_grad():
    err = (net(xs) - torch.sin(math.pi * xs)).abs().max().item()
    ends = net(torch.tensor([[0.0], [1.0]]))[:, 0].tolist()
print(f"step 499: residual loss {res_loss.item():.3f}, max error {err:.2f}")
print(f"u(0) = {ends[0]:.2f}, u(1) = {ends[1]:.2f}  (the exact solution has 0 at both ends)")
```

The residual loss fell about 900-fold, and the answer is still off by 0.9:
the ends sit away from 0. The network found another member of the family,
roughly `sin(pi x) + a x + b`.

# Task
- `d2dx2(u, x)`: the second derivative, from your `ddx`.
- Train the given `net` for 500 steps with Adam, lr 1e-2. Loss: the mean
  squared Poisson residual `u'' - f` on `x` plus the **boundary loss**
  `mean(net(xb)**2)` (both end values should be 0). Append each step's total
  loss as a float to `losses`.
- `max_err`: the largest `|net(x) - sin(pi x)|` on `x_test`, under
  `torch.no_grad()`.

# Given
```python
torch.set_num_threads(1)
x = torch.linspace(0, 1, 64)[:, None].requires_grad_()   # collocation points
f = -math.pi ** 2 * torch.sin(math.pi * x.detach())      # source term at x
xb = torch.tensor([[0.0], [1.0]])                        # boundary points
x_test = torch.linspace(0, 1, 201)[:, None]
torch.manual_seed(0)
net = MLP()
```

# Starter
```python
def d2dx2(u, x):
    TODO

opt = torch.optim.Adam(net.parameters(), lr=1e-2)
losses = []
for step in range(500):
    opt.zero_grad()
    # TODO: residual loss + boundary loss, backward, step, record
max_err = TODO
```

# Solution
```python
def d2dx2(u, x):
    return ddx(ddx(u, x), x)

opt = torch.optim.Adam(net.parameters(), lr=1e-2)
losses = []
for step in range(500):
    opt.zero_grad()
    res_loss = ((d2dx2(net(x), x) - f) ** 2).mean()
    bc_loss = (net(xb) ** 2).mean()
    loss = res_loss + bc_loss
    loss.backward()
    opt.step()
    losses.append(loss.item())
with torch.no_grad():
    max_err = (net(x_test) - torch.sin(math.pi * x_test)).abs().max().item()
print(f"step 499: loss {losses[-1]:.2e}, max error {max_err:.2e}")
```

# Checks
```python
xc = torch.linspace(-1, 1, 9)[:, None].requires_grad_()
check("d2dx2(sin(3x)) = -9 sin(3x)", d2dx2(torch.sin(3 * xc), xc), want=-9 * torch.sin(3 * xc.detach()), shape=(9, 1), atol=1e-4)
expect("d2dx2 keeps its graph", d2dx2(torch.sin(3 * xc), xc).grad_fn is not None, "the result has no grad_fn, so the loss cannot be trained through it: create_graph=True on both calls")
with torch.no_grad():
    u_test = net(x_test)
    true_err = (u_test - torch.sin(math.pi * x_test)).abs().max().item()
    ends = net(xb)[:, 0].tolist()
    mid = net(torch.tensor([[0.5]])).item()
expect("at most 1000 steps, one loss per step", 0 < len(losses) <= 1000, f"{len(losses)} entries in losses")
expect("u(0) and u(1) are about 0", max(abs(v) for v in ends) < 0.05, f"u(0) = {ends[0]:.3f}, u(1) = {ends[1]:.3f}. Is the boundary loss in the loss you call backward() on?")
expect("u(0.5) is about +1", abs(mid - 1) < 0.1, f"u(0.5) = {mid:.3f}, want about 1." + (" It is about -1: the residual has the wrong sign, use u'' - f." if mid < -0.5 else ""))
expect("max error < 0.05", true_err < 0.05, f"max |net - sin(pi x)| = {true_err:.3g}. If the loss is stuck near 48, the second derivative is treated as a constant: is create_graph=True on both calls?")
check("max_err", max_err, want=true_err, atol=1e-6)
```

# Hints
1. `d2dx2` is one line. The residual compares `d2dx2(net(x), x)` with `f`.
2. The boundary loss uses `net(xb)`, not `net(x)`: two points, both should be 0.

# Takeaway
500 steps took about 2 seconds, max error 0.0004-0.013. Leaving out the
boundary term gave the family member of the demo, with the ends away from 0.

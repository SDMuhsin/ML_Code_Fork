---
title: Second order
difficulty: Medium
uses: [2j-1]
inspect: [jet1d]
---
# Lesson
## Carrying u'' too

Add a third channel, `h''`. At the input `x'' = 0`. A Linear layer applies
its weight to it, with no bias, as to `h'`. For tanh, write `f = tanh`, so
`t = f(h)`, and differentiate `t' = f'(h) h'` once more by the product rule:

    t'' = f''(h) * h'**2 + f'(h) * h''

with `f'(h) = 1 - t**2` and `f''(h) = -2 t (1 - t**2)`.

The first term is easy to forget. At the first hidden layer `h''` is 0
(`h = x @ W.T + b` is a straight line in `x`), so without that term every
`h''` stays 0. The demo shows it on one tanh unit, `t = tanh(3x)`:

```python demo
x = torch.linspace(-1, 1, 5, dtype=torch.float64)[:, None]
h, h1, h2 = 3 * x, 3 * torch.ones_like(x), torch.zeros_like(x)
t = torch.tanh(h)
d1, d2 = 1 - t * t, -2 * t * (1 - t * t)
xr = x.clone().requires_grad_()
auto = ddx(ddx(torch.tanh(3 * xr), xr), xr)
print("autograd t'':           ", [round(v, 4) + 0.0 for v in auto[:, 0].tolist()])
print("f'' h'**2 + f' h'':     ", [round(v, 4) + 0.0 for v in (d2 * h1 * h1 + d1 * h2)[:, 0].tolist()])
print("f' h'' alone (wrong):   ", [round(v, 4) + 0.0 for v in (d1 * h2)[:, 0].tolist()])   # + 0.0 turns -0.0 into 0.0
```

A jet of order 2, the value and its first two derivatives, is called a
**Taylor jet** (they are the first terms of the Taylor series at the point).
The code in `pinn-solver-accel` calls its forward "Taylor mode".

The checks also print the step's FLOPs and saved memory next to nested
autograd. For one output the jet saves about as much memory; that is
expected, and `2j-3` shows the case where it saves half.

# Task
`jet1d(net, x)`: return `(u, u_x, u_xx)` for any `demo_net(din=1, ...)`, each
`(N, 1)`, by carrying three channels. Start from your `jet1` (available here
from `2j-1`) and add the third channel.

# Starter
```python
def jet1d(net, x):
    h, h1, h2 = x, torch.ones_like(x), torch.zeros_like(x)
    for m in net:
        if isinstance(m, nn.Linear):
            TODO
        else:
            TODO
    return h, h1, h2
```

# Solution
```python
def jet1d(net, x):
    h, h1, h2 = x, torch.ones_like(x), torch.zeros_like(x)
    for m in net:
        if isinstance(m, nn.Linear):
            h, h1, h2 = F.linear(h, m.weight, m.bias), h1 @ m.weight.T, h2 @ m.weight.T
        else:
            t = torch.tanh(h)
            d1 = 1 - t * t
            d2 = -2 * t * d1
            h, h1, h2 = t, d1 * h1, d2 * h1 * h1 + d1 * h2
    return h, h1, h2
```

# Checks
```python
torch.manual_seed(2)
xs = torch.linspace(-1, 1, 9, dtype=torch.float64)[:, None]
for depth in (1, 3):
    cnet = demo_net(depth=depth).double()
    cu, cux, cuxx = jet1d(cnet, xs)
    xr = xs.clone().requires_grad_()
    a_u = cnet(xr); a_ux = ddx(a_u, xr); a_uxx = ddx(a_ux, xr)
    check(f"u, depth {depth}", cu, want=a_u, shape=(9, 1), atol=1e-10)
    check(f"u_x, depth {depth}", cux, want=a_ux, shape=(9, 1), atol=1e-10)
    expect(f"u_xx is not all 0, depth {depth}", cuxx.abs().max().item() > 1e-6, "u_xx is 0 everywhere: the f''(h) * h'**2 term is missing (h'' starts at 0)")
    check(f"u_xx = ddx(ddx(u)), depth {depth}", cuxx, want=a_uxx, shape=(9, 1), atol=1e-10)
torch.manual_seed(0)
rnet = demo_net()
rx = torch.rand(1000, 1)
F1 = flops(lambda: run_net(rnet, rx))[0]
jet_loss = lambda: (jet1d(rnet, rx)[2] ** 2).mean()
fl = flops(lambda: jet_loss().backward())[0]
sv = count_saved(jet_loss)[1] / (1000 * 32 * 4)
print(f"your jet step: {fl / F1:.1f}F, saved {sv:.1f} A   (nested: 11.8F, 26.2 A)")
```

# Hints
1. Linear: the same `@ m.weight.T` on `h2` as on `h1`, no bias.
2. tanh: `d2 * h1 * h1 + d1 * h2`, with `h1` the derivative before this layer.

# Takeaway
Three channels gave `u_xx` exactly (1e-16 in float64), with no derivative
pass. The checks printed the cost of the step `mean(u_xx**2)` on the
reference network of `2g`: 8.9F and 27.2 A for the jet, against 11.8F and
26.2 A for nested `ddx(ddx(u))`. The jet does each matrix product once per
channel forward (3F) and twice that in the weight backward (6F). It saves
about as much as nested autograd, because every elementwise operation on
every channel saves its inputs for the weight backward.
On one CPU thread the jet step took 0.88-0.91 of the nested step's time
over two runs. That is not much for one output; the jet pays when the
network has several outputs.

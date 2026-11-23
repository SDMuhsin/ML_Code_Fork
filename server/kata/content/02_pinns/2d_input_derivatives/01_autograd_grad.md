---
title: autograd.grad with respect to x
difficulty: Easy
inspect: [ddx]
---
# Lesson
## How does u change when x moves?

A physics law such as "the rate of change of `u` in time is `-2u`" is a
statement about `du/dt`. Below the input is called `x`, as in most PDE code. Autograd computes that **input derivative**
the same way it computes a weight gradient. Mark the input `x` as tracked,
compute `u` from it, and ask for the derivative with
`torch.autograd.grad(u, x)`. It returns a tuple with one entry per tensor
you asked about, and it leaves `.grad` alone.

Try it on `u = sin(3x)` at 5 points:

```python demo
x = torch.linspace(0, 1, 5)[:, None].requires_grad_()
u = torch.sin(3 * x)
try:
    torch.autograd.grad(u, x)
except RuntimeError as e:
    print("error:", e)
```

The error says autograd can differentiate only a single number, and `u`
holds 5. The argument `grad_outputs` tells it which single number to use. It
is a tensor shaped like `u`, and autograd differentiates the weighted sum `(grad_outputs * u).sum()` instead. With
weight 1 on point 0 and 0 elsewhere you get the derivative of `u[0]` alone:

```python demo
v = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]])
print(torch.autograd.grad(u, x, grad_outputs=v)[0].T)
```

`3 cos(0) = 3`, at position 0, and zeros elsewhere because `u[0]` does not
depend on `x[1]` ... `x[4]`. With all weights 1 the sum is `u[0] + ... + u[4]`.
Each `u[i]` depends only on `x[i]`, so the derivative of the sum at `x[i]` is
`du[i]/dx[i]`: one call gives the derivative at every point. Problem `2d-2`
shows why this holds and when it fails.

```python demo
u = torch.sin(3 * x)
du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u))[0]
print("shape:", S(du), "  max |du - 3 cos(3x)| =", (du - 3 * torch.cos(3 * x)).abs().max().item())
x_plain = torch.linspace(0, 1, 5)[:, None]
try:
    torch.autograd.grad(torch.sin(3 * x_plain), x_plain, torch.ones(5, 1))
except RuntimeError as e:
    print("x without requires_grad:", e)
```

The result has the shape of `x`. If `x` was not marked with
`requires_grad_()` before computing `u`, autograd has recorded nothing and
refuses.

# Task
`ddx(u, x)`: return du/dx at every point, shaped like `x`, using
`torch.autograd.grad` with `grad_outputs` of ones. No other arguments.
The caller has already marked `x` with `requires_grad_()` and computed `u`
from it. Later problems build on this function.

# Starter
```python
def ddx(u, x):
    TODO
```

# Solution
```python
def ddx(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u))[0]
```

# Checks
```python
xs = torch.linspace(-1, 1, 9, dtype=torch.float64)[:, None].requires_grad_()
check("ddx(sin(3x), x) = 3 cos(3x)", ddx(torch.sin(3 * xs), xs), want=3 * torch.cos(3 * xs.detach()), shape=(9, 1), atol=1e-6)
check("ddx(x**3, x) = 3 x**2", ddx(xs ** 3, xs), want=3 * xs.detach() ** 2, shape=(9, 1), atol=1e-6)
torch.manual_seed(0)
fnet = demo_net().double()
def central(f, x, h=1e-6):
    with torch.no_grad():
        return (f(x + h) - f(x - h)) / (2 * h)
check("ddx of an MLP vs finite differences", ddx(fnet(xs), xs), want=central(fnet, xs.detach()), shape=(9, 1), atol=1e-6)
expect("x.grad was not touched", xs.grad is None, "x.grad was filled: use torch.autograd.grad, not u.backward()")
```

# Hints
1. The error says the output must be one number. Which argument turns `u` into a weighted sum?
2. `torch.ones_like(u)` has the shape `grad_outputs` needs. `autograd.grad` returns a tuple.

# Takeaway
One call with `grad_outputs = ones` gave the derivative at every point; the
checks compared it with finite differences on an MLP and it agreed to 1e-6. Without `grad_outputs` the call
fails on any `u` with more than one number; without `requires_grad_()` on `x`
there is nothing to differentiate.

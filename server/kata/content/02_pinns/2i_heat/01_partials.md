---
title: Two inputs, partial derivatives
difficulty: Medium
uses: [2b-2, 2d-3]
inspect: [heat_residual]
---
# Lesson
## The equation

The **heat equation** in one space dimension is

    u_t = k u_xx,   k = 0.1,   x in [0, 1],   t in [0, 1]

`u(x, t)` is the temperature at position `x` along a rod at time `t`.
`u_t` is its rate of change in time and `u_xx` its second derivative in
space (the curvature along the rod); `k` sets how fast heat spreads. A point
colder than the average of its neighbours has `u_xx > 0` and warms up, so
bumps flatten out. The rod starts as `u(x, 0) = sin(pi x)` (the **initial
condition**, now a whole row of values) and its ends are held at 0
(`u(0, t) = u(1, t) = 0`). The exact solution keeps the shape and shrinks it:
`u = exp(-k pi**2 t) sin(pi x)`.

## How do you get u_t and u_xx from one network?

The network now takes two numbers per point: `xt` has shape `(N, 2)`, column
0 is `x` and column 1 is `t`. `ddx(u, xt)` returns a tensor shaped like its
input, `(N, 2)`. Entry `[i, j]` is the derivative of `u[i]` with respect to
`xt[i, j]` (points do not interact, as in `2d-2`). So column 0 is `u_x` and
column 1 is `u_t`, both from one backward pass. The demo checks this on
`u = x**2 t`, where `u_x = 2 x t`, `u_t = x**2`, `u_xx = 2 t`, `u_xt = 2 x`:

```python demo
xt = torch.tensor([[0.5, 1.0], [1.0, 2.0], [2.0, 3.0]], requires_grad=True)
x, t = xt[:, 0:1], xt[:, 1:2]
u = x ** 2 * t
g = ddx(u, xt)
print("shape of ddx(u, xt):", S(g))
print("column 0:", g[:, 0].tolist(), " 2 x t:", (2 * x * t)[:, 0].tolist())
print("column 1:", g[:, 1].tolist(), " x**2: ", (x ** 2)[:, 0].tolist())
u_x = g[:, 0:1]
g2 = ddx(u_x, xt)
print("ddx(u_x, xt) column 0:", g2[:, 0].tolist(), " 2 t:", (2 * t)[:, 0].tolist())
print("ddx(u_x, xt) column 1:", g2[:, 1].tolist(), " 2 x:", (2 * x)[:, 0].tolist())
```

Slice with `g[:, 0:1]`, which keeps shape `(N, 1)`. `g[:, 0]` has shape `(N,)`, and
`(N,)` against `(N, 1)` broadcasts to `(N, N)`.

`u_xx` is column 0 of the second call. Its column 1 is the mixed derivative
`u_xt`, which the heat equation does not use. A `grad` call always returns
the derivative with respect to the whole tensor `xt`, so it is computed and
thrown away. It costs little: the two columns share the whole backward pass
and differ only in the last product, with the first layer's `(width, 2)`
weight.

# Task
`heat_residual(u_fn, xt)`: the residual `u_t - K u_xx` at the points `xt`
(shape `(N, 2)`, already requiring grad), shape `(N, 1)`. `u_fn` maps
`(N, 2)` to `(N, 1)`. Use your `ddx`; `K` is in Given.

# Given
```python
K = 0.1
xt = torch.rand(20, 2, generator=gen(0)).requires_grad_()
```

# Starter
```python
def heat_residual(u_fn, xt):
    TODO
```

# Solution
```python
def heat_residual(u_fn, xt):
    u = u_fn(xt)
    g = ddx(u, xt)
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = ddx(u_x, xt)[:, 0:1]
    return u_t - K * u_xx
```

# Checks
```python
xc, tc = xt[:, 0:1].detach(), xt[:, 1:2].detach()
exact = lambda z: torch.exp(-K * math.pi ** 2 * z[:, 1:2]) * torch.sin(math.pi * z[:, 0:1])
r_exact = heat_residual(exact, xt)
expect("residual shape (20, 1)", S(r_exact) == (20, 1), f"residual has shape {S(r_exact)}" + (": take columns with 0:1 (g[:, 0:1]), not g[:, 0]" if S(r_exact) == (20, 20) else ""))
check("exact solution has residual 0", r_exact, want=torch.zeros(20, 1), atol=1e-4)
wrong_k = lambda z: torch.exp(-0.2 * math.pi ** 2 * z[:, 1:2]) * torch.sin(math.pi * z[:, 0:1])
check("residual of a solution with k = 0.2 is -0.1 pi**2 u", heat_residual(wrong_k, xt), want=-0.1 * math.pi ** 2 * wrong_k(xt.detach()), shape=(20, 1), atol=1e-4)
poly = lambda z: z[:, 0:1] ** 3 * z[:, 1:2] + z[:, 1:2] ** 2
r_poly = heat_residual(poly, xt)
want_poly = xc ** 3 + 2 * tc - K * 6 * xc * tc
mixed = xc ** 3 + 2 * tc - K * (6 * xc * tc + 3 * xc ** 2)
same("residual shape for u = x**3 t + t**2", S(r_poly), (20, 1))
expect("u = x**3 t + t**2: residual x**3 + 2t - 6 K x t", torch.allclose(r_poly.detach().double().cpu(), want_poly.double().cpu(), atol=1e-4), "wrong value" + (": it matches u_t - K (u_xx + u_xt). The second call returns [u_xx, u_xt]; keep only column 0." if torch.allclose(r_poly.detach().double().cpu(), mixed.double().cpu(), atol=1e-4) else ". Column 0 of ddx(u, xt) is u_x, column 1 is u_t; u_xx is column 0 of ddx(u_x, xt)."))
expect("the residual keeps its graph", r_poly.grad_fn is not None, "no grad_fn: the loss could not be trained through it. Does ddx use create_graph=True?")
```

# Hints
1. Two `ddx` calls. The first gives `u_x` and `u_t` together.
2. Keep shape `(N, 1)` when you take a column: `g[:, 0:1]`.

# Takeaway
One backward pass gives the derivative with respect to every input column,
so `u_t` came free with `u_x`. Only `u_xx` needed a second pass. In 2D or 3D
space, `u_xx + u_yy (+ u_zz)` needs one second pass per space coordinate
(`2g-3`), because each pass starts from one column of the first result.

---
title: Counting FLOPs
difficulty: Medium
inspect: [matmul_flops, predict_total]
---
# Lesson
## Why is a backward pass about twice a forward pass?

A **FLOP** is one floating-point multiply or one add. A matrix product of an
`(N, k)` matrix with a `(k, m)` matrix makes `N * m` outputs, each `k`
multiplies and `k` adds: `2 * N * k * m` FLOPs. In an MLP almost all the
FLOPs are in these products. PyTorch's `FlopCounterMode` counts the FLOPs of
matrix products only (tanh, add and the like are not counted); `flops(fn)` in
Shared wraps it. In this problem "FLOPs" means matrix-product FLOPs.

A Linear layer `y = h @ W.T` does one product forward. Its weight backward
does two of the same size: `g.T @ h` for the weight gradient and `g @ W` for
the gradient sent on to the layer below:

```python demo
torch.manual_seed(0)
net = demo_net()
x = torch.linspace(0, 1, 1000)[:, None]
f_fwd, loss = flops(lambda: pinn_loss(net, x, 0))
f_bwd, _ = flops(lambda: loss.backward())
print(f"forward {f_fwd / 1e6:.2f} MFLOP, backward {f_bwd / 1e6:.2f} MFLOP, ratio {f_bwd / f_fwd:.2f}")
```

(MFLOP = one million FLOPs.) The ratio is slightly under 2 because the first
layer sends no gradient to `x`.

## And with input derivatives?

Two rules give every number below. Call the forward's FLOPs F.

- A derivative pass toward `x` does **one** product for each product in the
  graph it walks: in `h @ W.T` only `h` depends on `x`, so it needs only
  `g @ W`. It costs as many FLOPs as the graph it walks: the `u_x` pass walks
  the forward (F) and adds F, total 2F; the `u_xx` pass walks that 2F graph
  and adds 2F, total 4F.
- The weight backward does **two** products for each product in the graph it
  walks, so it costs twice that graph: 2F at order 0, 4F at order 1, 8F at
  order 2.

```python demo
for order in range(3):
    f1, loss = flops(lambda: pinn_loss(net, x, order))
    f2, _ = flops(lambda: loss.backward())
    print(f"order {order}: forward + derivatives {f1 / 1e6:6.2f}, backward {f2 / 1e6:6.2f}, total {(f1 + f2) / 1e6:6.2f} MFLOP")
f32, _ = flops(lambda: pinn_loss(net, x, 0).backward())
f64, _ = flops(lambda: pinn_loss(demo_net(width=64), x, 0).backward())
print(f"order 0, width 32: {f32 / 1e6:.2f} MFLOP, width 64: {f64 / 1e6:.2f} MFLOP ({f64 / f32:.1f} times)")
```

The input derivative alone is cheaper than a weight backward (one product
per layer instead of two). The cost comes from the weight backward walking
the longer graph that `create_graph=True` recorded.

# Task
- `matmul_flops(N, width, depth)`: the forward FLOPs of `demo_net(width=width,
  depth=depth)` on `N` points from the `2 * N * k * m` rule. The layers are
  `1 -> width`, then `depth - 1` times `width -> width`, then `width -> 1`.
- `predict_total(N, width, depth, order)`: FLOPs of forward, derivatives and
  backward together, from `matmul_flops` and the two rules.

Both return numbers. The checks compare them with `flops` on other networks.

# Starter
```python
def matmul_flops(N, width, depth):
    TODO

def predict_total(N, width, depth, order):
    TODO
```

# Solution
```python
def matmul_flops(N, width, depth):
    return 2 * N * (width + (depth - 1) * width * width + width)

def predict_total(N, width, depth, order):
    return 3 * 2 ** order * matmul_flops(N, width, depth)
```

# Checks
```python
def measured(N, width, depth, order):
    torch.manual_seed(0)
    cnet = demo_net(width=width, depth=depth)
    cx = torch.rand(N, 1)
    f1, closs = flops(lambda: pinn_loss(cnet, cx, order))
    f2, _ = flops(lambda: closs.backward())
    return f1, f1 + f2
for N, width, depth in [(500, 64, 4), (300, 16, 2)]:
    fwd = measured(N, width, depth, 0)[0]
    got = matmul_flops(N, width, depth)
    expect(f"matmul_flops({N}, {width}, {depth}) = {fwd}", got == fwd, f"got {got}, want {fwd} ({got / fwd:.2f} times). The first and last layers are 1 -> width and width -> 1, not width x width.")
for order in range(4):
    fwd, tot = measured(500, 64, 4, order)
    p = predict_total(500, 64, 4, order)
    expect(f"predict_total(500, 64, 4, {order}) within 5% of {tot / 1e6:.2f} MFLOP", abs(p - tot) <= 0.05 * tot, f"predicted {p / 1e6:.2f} MFLOP, measured {tot / 1e6:.2f}. As multiples of the forward: yours {p / matmul_flops(500, 64, 4):.2f}, measured {tot / measured(500, 64, 4, 0)[0]:.2f}. If these agree, fix matmul_flops first.")
```

# Hints
1. `matmul_flops`: three kinds of layer, each `2 * N * (in features) * (out features)`.
2. `predict_total`: at order 0 the total is F + 2F. Write forward-plus-derivatives and backward for order 1 in units of F, then order 2.

# Takeaway
Matrix-product FLOPs double with each derivative order: 3F, 6F, 12F for
orders 0-2, with F = 4.2 MFLOP on the reference network. In one formula, for
width well above 1:

    FLOPs  ~ 3 * 2**order * 2 * N * width**2 * depth

so FLOPs grow with the square of the width, but linearly with points and
depth. Saved memory (`2f-2`) grows only linearly with the width: doubling the
width costs 4 times the FLOPs and 2 times the memory. The next problem shows
that saved memory grows faster than x2 per order.

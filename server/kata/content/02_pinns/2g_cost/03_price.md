---
title: Price a step before you run it
difficulty: Hard
devices: [cuda, cpu]
inspect: [flops_F, saved_A2, peak_A2, max_points]
timeout: 60
---
# Lesson
## Measuring live and peak memory

A GPU reports memory directly, with its peak. `torch.cuda.memory_allocated()` is the bytes
held by live tensors now. `torch.cuda.max_memory_allocated()` is the highest
value since `torch.cuda.reset_peak_memory_stats()`; `peak_A` in Shared wraps
both. Live memory after each stage of an order-2 step, and after a `u_x`
without `create_graph`:

```python demo
A = 1000 * 32 * 4
if DEVICE == "cuda":
    torch.manual_seed(0)
    net = demo_net()
    x = torch.linspace(0, 1, 1000)[:, None]
    pinn_loss(net, x, 2).backward(); net.zero_grad(set_to_none=True)   # run once first, not measured
    live = lambda: (torch.cuda.synchronize(), torch.cuda.memory_allocated())[1]
    base, xr = live(), x.detach().requires_grad_()
    u = net(xr); a0 = (live() - base) / A
    ux = torch.autograd.grad(u, xr, torch.ones_like(u), create_graph=True)[0]; a1 = (live() - base) / A
    uxx = torch.autograd.grad(ux, xr, torch.ones_like(ux), create_graph=True)[0]; a2 = (live() - base) / A
    del u, ux, uxx
    base = live(); u = net(xr)
    ux = torch.autograd.grad(u, xr, torch.ones_like(u))[0]; a3 = (live() - base) / A
else:
    print("(no GPU in this run: numbers measured on an RTX 3050 Ti)")
    a0, a1, a2, a3 = 3.03, 9.10, 26.16, 0.06
print(f"after u: {a0:.1f} A   after u_x: {a1:.1f} A   after u_xx: {a2:.1f} A   u_x without create_graph: {a3:.2f} A")
```

The live memory equals the saved values of `2g-2`. A derivative pass
without `create_graph` frees what it walked; with it, nothing is freed until
`loss.backward()` walks the graph. Until then, a graph tensor you
hold, such as `u_x` or an old `loss`, keeps its part alive.

The **peak** comes inside the weight backward, which creates `(N, width)`
gradients before it has freed the saved values.

## Cost table for the reference network

1000 points. F is the forward's matrix-product FLOPs
(4.2 MFLOP).

| step | FLOPs | saved | peak |
|---|---|---|---|
| plain | 3F | 3.1 A | 5.0 A |
| residual with `u_x` | 6F | 9.2 A | 13.0 A |
| residual with `u_xx` | 12F | 26.2 A | 29.1 A |

The peak is 2 to 4 A above the saved memory, so for pricing take **peak =
saved + 3 A**. FLOPs double per order and, with tanh, memory triples
(`2g-2`). Both are linear in `N` on any device. This lesson counts FLOPs and
bytes rather than time because time depends on the device: on a GPU this
small network is limited by the fixed cost of starting each operation, so its
step time barely changes with `N`.

## Sibling passes

The rules so far are for a **chain**: the `u_xx` pass walks the graph that
the `u_x` pass recorded. With two inputs, the `u_xx` pass (from `u_x`) and
the `u_yy` pass (from `u_y`) both walk the order-1 graph and neither walks
the other: they are **sibling passes**. Each costs what one order-2 pass costs on that graph;
the costs add, and so does the memory each adds. The weight backward walks
everything recorded, at twice its FLOPs.

# Task
Price a step you have not run: the **2D Poisson** residual
`u_xx + u_yy - f` on `demo_net(din=2)` (input `(N, 2)`, columns `x` and `y`),
1000 points. One derivative pass on the `(N, 2)` input gives `u_x` and `u_y`
together, as its two columns. `u_xx` needs a second pass from `u_x`, and `u_yy`
another one from `u_y`. With F the forward FLOPs of this network and A =
125 KB as before, set:

- `flops_F`: forward, derivative passes and weight backward together, in
  units of F;
- `saved_A2`: the saved memory, in A, from the per-layer counts of `2g-2`;
- `peak_A2`: the peak, in A (on a CPU run the checks compare with the peak
  measured on an RTX 3050 Ti);
- `max_points`: the largest `N` (an int) whose peak fits in 3 GB = 3072 MB
  (the rest of the 4 GB is for PyTorch and the display).

Work them out before you run anything. The checks measure the step and print
your numbers next to the measured ones.

# Starter
```python
flops_F = TODO
saved_A2 = TODO
peak_A2 = TODO
max_points = TODO
```

# Solution
```python
flops_F = 18        # forward F, first pass F, u_xx pass 2F, u_yy pass 2F; backward 2 * 6F
saved_A2 = 3 * 15   # per layer: 1 forward, 2 first pass, 6 for each of the two second passes
peak_A2 = saved_A2 + 3
max_points = int(3072 * 2**20 / (peak_A2 * 32 * 4))
```

# Checks
```python
torch.manual_seed(0)
pnet = demo_net(din=2)
px = torch.rand(1000, 2)
A = 1000 * 32 * 4
def lap_loss():
    x = px.detach().requires_grad_()
    u = run_net(pnet, x)
    g = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], x, torch.ones_like(u), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(g[:, 1:2], x, torch.ones_like(u), create_graph=True)[0][:, 1:2]
    return ((uxx + uyy) ** 2).mean()
F2 = flops(lambda: run_net(pnet, px))[0]
m_flops = flops(lambda: lap_loss().backward())[0] / F2
m_saved = count_saved(lap_loss)[1] / A
if DEVICE == "cuda":
    def lap_step():
        lap_loss().backward()
        pnet.zero_grad(set_to_none=True)
    m_peak = peak_A(lap_step, A)
else:
    m_peak = 46.3   # measured on an RTX 3050 Ti
m_max = 3072 * 2**20 / (m_peak * 32 * 4)
print(f"measured: {m_flops:.1f} F, saved {m_saved:.1f} A, peak {m_peak:.1f} A, max N about {m_max:,.0f}")
expect(f"flops_F within 10% of {m_flops:.1f}", not is_blank(flops_F) and abs(flops_F - m_flops) <= 0.1 * m_flops, f"you said {flops_F}. Before the weight backward: forward F, first pass F, and each second pass walks the 2F graph of u and its first derivatives. The weight backward is twice all of that.")
expect(f"saved_A2 within 15% of {m_saved:.1f} A", not is_blank(saved_A2) and abs(saved_A2 - m_saved) <= 0.15 * m_saved, f"you said {saved_A2}. A 1D order-2 step keeps 9 per layer: 3 after the first pass, 6 added by the second. Here two second passes walk the same order-1 graph.")
expect(f"peak_A2 within 15% of {m_peak:.1f} A", not is_blank(peak_A2) and abs(peak_A2 - m_peak) <= 0.15 * m_peak, f"you said {peak_A2}. In the table the peak was the saved memory plus about 3 A of backward buffers.")
expect(f"max_points within 15% of {m_max:,.0f}", not is_blank(max_points) and abs(max_points - m_max) <= 0.15 * m_max, f"you said {max_points}. One A is N * 32 * 4 bytes, so the peak is peak_A2 * N * 128 bytes; solve for N at 3072 * 2**20 bytes.")
```

# Hints
1. Each second-derivative pass walks the graph of `u` and its first derivatives, which cost 2F, and keeps what the `u_xx` pass of `2g-2` kept per layer.
2. For `max_points`, write the peak in bytes as a function of `N` first.

# Takeaway
The 2D Poisson step costs 18F, against 12F for the 1D one, and about 43 A
saved, against 26 A: each extra space dimension adds one second-derivative
pass, 6F of FLOPs (a 2F pass plus 4F of weight backward) and about 6 A per
hidden layer. On the
4 GB card about 540,000 points fit in one step on this network. You rarely
need that many in one step: the collocation points are free (`2e-2`), so use
fewer per step, draw new ones each step, and run more steps. Memory per step
falls with `N`, and over the run the network still sees many different points.

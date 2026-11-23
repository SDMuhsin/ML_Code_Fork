---
title: Five channels, three outputs
difficulty: Hard
uses: [2j-2]
inspect: [jet2d]
timeout: 30
---
# Lesson
## Which channels does a 2D residual need?

A PINN with input `(x, y)` whose residual reads `u_x`, `u_y`, `u_xx` and
`u_yy` needs five channels: the value, one per first derivative and one per
second derivative the residual reads.

| channel | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| meaning | value | d/dx | d/dy | d2/dx2 | d2/dy2 |

The mixed derivative `u_xy` is not read here, so it gets no channel. A
residual that reads it would add one more channel, with the tanh rule
`t_xy = f''(h) h_x h_y + f'(h) h_xy`, starting from `h_xy = 0` at the first
Linear layer. (The code in
`pinn-solver-accel` is 3D and reads mixed ones: 1 + 3 + 6 = 10 channels.)

The rules of `2j-2` apply channel by channel. Only the input changes. The
first Linear layer gets the `(N, 2)` input, so `h = xy @ W.T + b`. Its
derivatives are columns of the weight, the same at every point:
`h_x = W[:, 0]` and `h_y = W[:, 1]`. Its second derivatives are 0. In the tanh rule each
second-derivative channel uses its own first-derivative channel:
`t_xx = f'' * h_x**2 + f' * h_xx`, and the same with `y`.

## Several outputs at once

With `dout = 3` the last Linear layer has 3 rows, so every channel comes out
with 3 columns: the derivatives of all 3 outputs, from one forward pass. Nested
autograd needs separate derivative passes for each output, because a
backward pass differentiates one number (`2d-2`). For 3 outputs, each with
`u_xx` and `u_yy`, that is 9 derivative passes:

```python demo
torch.manual_seed(0)
net3 = demo_net(din=2, dout=3)
xy = torch.rand(1024, 2)
calls = []
plain_ddx = ddx
def counting_ddx(u, x):
    calls.append(1)
    return plain_ddx(u, x)
ddx = counting_ddx             # nested_2d in Shared now counts its derivative passes
nested_2d(net3, xy)
ddx = plain_ddx
print("derivative passes for 3 outputs:", len(calls))
```

A residual in `2m` reads 13 fields of a 3-output network (`u`, `v`, `p` and
their derivatives). For a step whose loss reads those 13 values, on
`demo_net(din=2, dout=3)` at 1024 points, with the `jet2d` you are about to
write (the checks print yours):

| | FLOPs | saved |
|---|---|---|
| nested autograd | 34.7F | 90 A |
| jet | 14.6F | 43 A |

On one CPU thread the nested step took 1.8-2.5 times as long as the jet
step (several runs; your times will differ). In `2j-2` the jet saved about as
much as nested autograd, which made 2 derivative passes for one output. Here
nested autograd records 7 derivative passes for the 13 values (9 if `p_xx`
and `p_yy` were read). The jet still records one forward of 5 channels, so
it saves about half.

# Task
`jet2d(net, xy)`: for `net = demo_net(din=2, dout=K)` and `xy` of shape
`(N, 2)`, return a tensor of shape `(5, N, K)`: channel `c` of output `k` at
point `i` is `J[c, i, k]`, in the channel order of the table above. Keep the
channels in a Python list of 5 tensors while you go through the layers, and
`torch.stack` them at the end.

# Starter
```python
def jet2d(net, xy):
    first = net[0]
    h = F.linear(xy, first.weight, first.bias)
    hx = torch.zeros_like(h) + first.weight[:, 0]     # the same row at every point
    hy = TODO
    hs = [h, hx, hy, torch.zeros_like(h), torch.zeros_like(h)]
    for m in list(net)[1:]:
        if isinstance(m, nn.Linear):
            TODO
        else:
            TODO
    return torch.stack(hs)
```

# Solution
```python
def jet2d(net, xy):
    first = net[0]
    h = F.linear(xy, first.weight, first.bias)
    hx = torch.zeros_like(h) + first.weight[:, 0]     # the same row at every point
    hy = torch.zeros_like(h) + first.weight[:, 1]
    hs = [h, hx, hy, torch.zeros_like(h), torch.zeros_like(h)]
    for m in list(net)[1:]:
        if isinstance(m, nn.Linear):
            hs = [F.linear(hs[0], m.weight, m.bias)] + [c @ m.weight.T for c in hs[1:]]
        else:
            t = torch.tanh(hs[0])
            d1 = 1 - t * t
            d2 = -2 * t * d1
            hs = [t, d1 * hs[1], d1 * hs[2], d2 * hs[1] * hs[1] + d1 * hs[3], d2 * hs[2] * hs[2] + d1 * hs[4]]
    return torch.stack(hs)
```

# Checks
```python
torch.manual_seed(3)
cxy = torch.rand(16, 2, dtype=torch.float64) * 2 - 1
names = ["value", "d/dx", "d/dy", "d2/dx2", "d2/dy2"]
for K in (1, 3):
    cnet = demo_net(din=2, dout=K).double()
    J = jet2d(cnet, cxy)
    same(f"shape for K = {K}", S(J), (5, 16, K))
    ref = nested_2d(cnet, cxy)
    for c in range(5):
        check(f"K = {K}, channel {c} ({names[c]})", J[c], want=torch.cat([ref[k][c] for k in range(K)], 1), atol=1e-10)
torch.manual_seed(0)
pnet = demo_net(din=2, dout=3)
pxy = torch.rand(1024, 2)
F1 = flops(lambda: run_net(pnet, pxy))[0]
def nested13(net, xy):
    # the 13 values of the table: no second derivatives of the third output (p)
    x = xy.detach().requires_grad_()
    out = run_net(net, x)
    cols = []
    for k in range(3):
        u = out[:, k:k + 1]
        g = ddx(u, x)
        cols += [u, g[:, 0:1], g[:, 1:2]]
        if k < 2:
            cols += [ddx(g[:, 0:1], x)[:, 0:1], ddx(g[:, 1:2], x)[:, 1:2]]
    return torch.cat(cols, 1)
def jet13(net, xy):
    J = jet2d(net, xy)
    return torch.cat([J[0], J[1], J[2], J[3, :, 0:2], J[4, :, 0:2]], 1)
A1 = 1024 * 32 * 4
fl_nest = flops(lambda: (nested13(pnet, pxy) ** 2).mean().backward())[0]
fl_jet = flops(lambda: (jet13(pnet, pxy) ** 2).mean().backward())[0]
sv_nest = count_saved(lambda: (nested13(pnet, pxy) ** 2).mean())[1] / A1
sv_jet = count_saved(lambda: (jet13(pnet, pxy) ** 2).mean())[1] / A1
print(f"13-value step: nested {fl_nest / F1:.1f}F, {sv_nest:.1f} A; your jet {fl_jet / F1:.1f}F, {sv_jet:.1f} A")
```

# Hints
1. Linear: the value channel gets `F.linear` with the bias; channels 1-4 get `c @ m.weight.T`.
2. tanh: channel 3 is `d2 * hs[1] * hs[1] + d1 * hs[3]`; channel 4 is the same with 2 and 4.
3. If only the `d2/dx2` channel is wrong for `K = 3` too, check which first-derivative channel it squares.

# Takeaway
Nested autograd repeats its derivative passes for each output, so its cost
grows with the number of outputs K. The jet's cost is set by the number of
channels: 5 channels forward (5F, a little less because the first layer's
derivative channels need no product) and twice that in the weight backward,
about 15F for any K. Timed with width 64, depth 4, 1024 points and all 5
channels of every output in the loss, on one CPU thread, nested autograd took
about 1.25, 3.3 and 7.3-7.4 times as long as this jet for K = 1, 3, 7. The
jet's own time changed by under 10% across the three.

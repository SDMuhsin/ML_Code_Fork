---
title: Kovasznay flow
difficulty: Hard
devices: [cpu]
uses: [2j-3, 2l-1, 2l-2, 2l-3]
timeout: 120
inspect: [history, max_err]
---
# Lesson
## The equations

A fluid moving in a plane has, at each point, a velocity `(u, v)` and a
pressure `p`: three outputs of one network. The flow here is **steady** (it
does not change in time) and **incompressible**: as much fluid flows into
any small box as flows out, which is `u_x + v_y = 0`. The **Navier-Stokes
equations** add one balance per velocity component:

    u u_x + v u_y + p_x - nu (u_xx + u_yy) = 0
    u v_x + v v_y + p_y - nu (v_xx + v_yy) = 0

The first two terms carry the velocity along with the flow, `p_x` pushes
fluid from high to low pressure, and the **viscosity** `nu` smooths the
velocity. `make_ns_loss(nu)` in Shared is the sum of the three residuals
squared, per point: the loss you traced in `2l`.

## A flow with a known answer

**Kovasznay flow** is an exact solution, the flow behind a row of
obstacles. With `Re = 1/nu = 40` (the **Reynolds number**; for this flow
larger means less viscous) and `lam = Re/2 - sqrt(Re**2/4 + 4 pi**2) = -0.964`:

    u = 1 - exp(lam x) cos(2 pi y)
    v = lam / (2 pi) exp(lam x) sin(2 pi y)
    p = (1 - exp(2 lam x)) / 2

`exact(xy)` in Shared computes it. The domain is `x` in `[0, 1]`, `y` in
`[-0.5, 0.5]`. The boundary loss is `((net(xb) - ub) ** 2).mean()`, over all 3
columns of the 160 edge points. The equations contain only `p_x` and `p_y`, so without the boundary
`p` would be fixed only up to a constant.

## One accelerated step: jet, emitted adjoint, one backward pass

1. `stack = ns_stack(jet2d(net, xy))`: the 13 fields, from your jet.
2. Under `torch.no_grad()`: the residual loss value for the history,
   `loss_point(as_dict(stack, NS_NAMES)).mean()`, and the adjoint stack
   `sage(stack, 1 / N)` from your `build_sage`.
3. `stack.backward(gradient=adj)`.
4. Boundary loss, its own `backward()`; the weight gradients of the two
   parts add up in `.grad` (`2a-2`).
5. `opt.step()` and `sched.step()`. `StepLR` multiplies the learning rate by
   0.3 every 500 steps.

# Task
Train the given `net` for 1500 steps with the loop above. Append
`(residual loss, boundary loss)` as Python floats to `history` each step.
Afterwards set `max_err` to a list of 3 floats: the largest `|net - exact|`
over `xy_test` for `u`, `v` and `p`: `err.max(0)` takes the maximum down
each column and returns `(values, indices)`, so use `.values.tolist()`. The
run takes well under a minute on one CPU thread; print `max_err` at the end
if you want to see it.

# Given
```python
torch.set_num_threads(1)
NU = 1 / RE
g = torch.linspace(0, 1, 20)
X, Y = torch.meshgrid(g, g - 0.5, indexing="ij")
xy = torch.stack([X.flatten(), Y.flatten()], 1)                       # (400, 2) collocation points
s = torch.linspace(0, 1, 40)
xb = torch.cat([torch.stack([0 * s, s - 0.5], 1), torch.stack([0 * s + 1, s - 0.5], 1),
                torch.stack([s, 0 * s - 0.5], 1), torch.stack([s, 0 * s + 0.5], 1)])   # (160, 2) edges
ub = exact(xb)                                                        # (160, 3)
gt = torch.linspace(0, 1, 51)
Xt, Yt = torch.meshgrid(gt, gt - 0.5, indexing="ij")
xy_test = torch.stack([Xt.flatten(), Yt.flatten()], 1)               # (2601, 2)
torch.manual_seed(0)
net = demo_net(din=2, dout=3)
```

# Starter
```python
loss_point = make_ns_loss(NU)
N = xy.shape[0]                   # number of collocation points
sage = TODO                       # your build_sage for make_ns_loss, NS_NAMES, NU
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.3)
history = []
for step in range(1500):
    opt.zero_grad()
    stack = TODO
    # TODO: residual loss value and adjoint stack (no graph), then backward through the jet
    # TODO: boundary loss and its backward; opt.step(); sched.step(); append (res, bc) floats
max_err = TODO
```

# Solution
```python
loss_point = make_ns_loss(NU)
sage = build_sage(make_ns_loss, NS_NAMES, NU)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.3)
history = []
N = xy.shape[0]
for step in range(1500):
    opt.zero_grad()
    stack = ns_stack(jet2d(net, xy))
    with torch.no_grad():
        res = loss_point(as_dict(stack, NS_NAMES)).mean()
        adj = sage(stack, 1 / N)
    stack.backward(gradient=adj)
    bc = ((net(xb) - ub) ** 2).mean()
    bc.backward()
    opt.step()
    sched.step()
    history.append((res.item(), bc.item()))
with torch.no_grad():
    max_err = (net(xy_test) - exact(xy_test)).abs().max(0).values.tolist()
print(f"step 1499: residual {history[-1][0]:.1e}, boundary {history[-1][1]:.1e}; max error u, v, p:", [round(e, 3) for e in max_err])
```

# Checks
```python
with torch.no_grad():
    true_err = (net(xy_test) - exact(xy_test)).abs().max(0).values.tolist()
last = history[-1] if len(history) else (float("nan"),) * 2
expect("history: one (res, bc) pair per step", len(history) == 1500 and all(len(h) == 2 for h in history), f"{len(history)} entries; want 1500 pairs")
expect("residual loss < 1e-3", last[0] < 1e-3, f"last residual loss {last[0]:.3g}. Does stack.backward(gradient=adj) run every step?")
expect("boundary loss < 1e-2", last[1] < 1e-2, f"last boundary loss {last[1]:.3g}. Does bc.backward() run every step? If it does, check the seed: 1 instead of 1 / N weights the residual N times more than the boundary, and the boundary is ignored (measured: boundary loss 0.098, max error of u 1.34).")
expect("max error of u < 0.2", true_err[0] < 0.2, f"max |u - exact| = {true_err[0]:.3g}; last residual, boundary loss {last[0]:.2g}, {last[1]:.2g}")
expect("max error of v < 0.05", true_err[1] < 0.05, f"max |v - exact| = {true_err[1]:.3g}")
expect("max error of p < 0.05", true_err[2] < 0.05, f"max |p - exact| = {true_err[2]:.3g}" + (". u and v fit, so does the boundary loss include the p column?" if true_err[0] < 0.2 and true_err[1] < 0.05 else ""))
check("max_err", torch.tensor(max_err), want=torch.tensor(true_err), atol=1e-6)
torch.manual_seed(0)
gnet = demo_net(din=2, dout=3)
gnet.zero_grad()
loss_point(as_dict(nested_stack(gnet, xy), NS_NAMES)).mean().backward()
g_nested = weight_grads(gnet).clone()
gnet.zero_grad()
gst = ns_stack(jet2d(gnet, xy))
gst.backward(gradient=build_sage(make_ns_loss, NS_NAMES, NU)(gst.detach(), 1 / xy.shape[0]))
gerr = rel_err(weight_grads(gnet), g_nested)
expect("accelerated step's weight gradient = nested autograd (residual part, step 0)", gerr < 1e-5, f"relative error {gerr:.3g}")
```

# Hints
1. Step 2 needs no graph: both lines go under one `with torch.no_grad():`.
2. The boundary loss is `((net(xb) - ub) ** 2).mean()` over all 3 columns.
3. `max_err`: `.abs().max(0).values.tolist()` gives the 3 column maxima.

# Takeaway
The network fitted three outputs from a residual with second derivatives,
with no `create_graph`. Over 7 random initializations (`torch.manual_seed(0)`
to `(6)`) the max errors were 0.07-0.13 for `u` (which ranges from 0 to 2),
at most 0.033 for `v` and 0.024 for `p`. The same 1500 steps with nested
autograd gave the same loss curve, because the gradients are equal. On one
CPU thread the nested run took 1.7-2.0 times as long as the accelerated run
(two runs; your times will differ). The next problem finds out which part of
the step that factor came from.

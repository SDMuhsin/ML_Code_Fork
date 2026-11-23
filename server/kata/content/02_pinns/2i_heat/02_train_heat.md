---
title: Train the heat PINN
difficulty: Medium
devices: [cpu]
uses: [2b-2, 2d-3, 2i-1]
timeout: 60
inspect: [history, max_err]
---
# Lesson
## Three conditions, three loss terms

The loss has three parts, each a mean squared miss on its own points:

- residual loss: `heat_residual` on the collocation points `xt`, a 16 x 16
  grid of `(x, t)` (256 points, edges included);
- initial loss: `net(x0)` against `sin(pi x)` on the row `t = 0` (16 points);
- boundary loss: `net(xb)` against 0 on the two columns `x = 0` and `x = 1`
  (32 points).

All three have weight 1; `2h-2` showed how to read the parts.

## What does one step cost?

The residual needs `u_t` and `u_xx`. `u_t` comes out of the same pass as
`u_x`, so the step has the passes of a Poisson step. The demo measures it on
1000 points, next to a plain step and a Poisson step:

```python demo
torch.manual_seed(0)
net1, net2 = demo_net(din=1), demo_net(din=2)
x1 = torch.rand(1000, 1).requires_grad_()
x2 = torch.rand(1000, 2).requires_grad_()
def heat_loss():
    u = run_net(net2, x2)
    g = ddx(u, x2)
    return ((g[:, 1:2] - 0.1 * ddx(g[:, 0:1], x2)[:, 0:1]) ** 2).mean()
steps = {"plain": lambda: (run_net(net1, x1.detach()) ** 2).mean(),
         "Poisson": lambda: (ddx(ddx(run_net(net1, x1), x1), x1) ** 2).mean(),
         "heat": heat_loss}
A = 1000 * 32 * 4
print("forward alone:", flops(lambda: run_net(net1, x1.detach()))[0] / 1e6, "MFLOP")
for name, loss_fn in steps.items():
    fl = flops(lambda: loss_fn().backward())[0]
    print(f"{name:8s} step: {fl / 1e6:5.1f} MFLOP, saved {count_saved(loss_fn)[1] / A:5.1f} A")
```

The rules of `2g` give 12F and 9 A per hidden layer, 27 A for three. The
demo measures about 12F and 26 A for the Poisson step and for the heat step:
the second input adds only one column to the first layer's product.

## When is a PINN worth it?

With one to three space dimensions, a simple shape, and all coefficients
and boundary values known, as in this lesson, a classic numerical solver is
faster and more accurate. The usual one is **finite differences**: keep `u`
only at grid points, replace each derivative by differences of neighbouring
values, and solve the linear equations, one per point, all at once. On
1000 points this solves the Poisson problem of `2e-3` to an error of about
1e-6 in under 1 ms of CPU time; the PINN reached 4e-4 to 1e-2 in 2 seconds.

A PINN is worth its cost in three cases.

- Part of the equation is unknown. Make the coefficient `k` a trainable
  parameter (in the optimizer's list) and add a data loss on a few measured
  values of `u`. The same step then fits the data and estimates `k`, if
  the measurements depend on `k` enough to pin it down. This **inverse
  problem** is one of the main uses of PINNs.
- One network for a family of problems: add the coefficient as an extra
  input, train once, and evaluate any member without solving again.
- Many input dimensions. A grid needs `n**d` points in `d` dimensions;
  collocation points do not grow that way.

Smaller benefits: no grid to fit an awkward shape, and derivatives
anywhere by autograd.

# Task
Train the given `net` for 800 steps with Adam, lr 1e-2, on the sum of the
three parts above, using your `heat_residual` (it reads `K` from Given). Each
step, append the three parts as Python floats to `history`, as a tuple
`(res, ic, bc)`. Afterwards set `max_err`: the largest `|net - exact|` on
the 41 x 41 grid `xt_test`, under `torch.no_grad()`. About 5 seconds.

# Given
```python
torch.set_num_threads(1)
K = 0.1
g = torch.linspace(0, 1, 16)
X, T = torch.meshgrid(g, g, indexing="ij")
xt = torch.stack([X.flatten(), T.flatten()], 1).requires_grad_()     # (256, 2) grid points, edges included
x0 = torch.stack([g, torch.zeros(16)], 1)                            # (16, 2) the row t = 0
u0 = torch.sin(math.pi * g)[:, None]                                 # initial values there
xb = torch.cat([torch.stack([torch.zeros(16), g], 1),
                torch.stack([torch.ones(16), g], 1)])                # (32, 2) x = 0 and x = 1
gt = torch.linspace(0, 1, 41)
Xt, Tt = torch.meshgrid(gt, gt, indexing="ij")
xt_test = torch.stack([Xt.flatten(), Tt.flatten()], 1)               # (1681, 2)
u_exact = torch.exp(-K * math.pi ** 2 * xt_test[:, 1:2]) * torch.sin(math.pi * xt_test[:, 0:1])
torch.manual_seed(0)
net = MLP(din=2)
```

# Starter
```python
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
history = []
for step in range(800):
    opt.zero_grad()
    res_loss = TODO
    ic_loss = TODO
    bc_loss = TODO
    # TODO: total loss, backward, step; append (res, ic, bc) floats to history
max_err = TODO
```

# Solution
```python
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
history = []
for step in range(800):
    opt.zero_grad()
    res_loss = (heat_residual(net, xt) ** 2).mean()
    ic_loss = ((net(x0) - u0) ** 2).mean()
    bc_loss = (net(xb) ** 2).mean()
    (res_loss + ic_loss + bc_loss).backward()
    opt.step()
    history.append((res_loss.item(), ic_loss.item(), bc_loss.item()))
with torch.no_grad():
    max_err = (net(xt_test) - u_exact).abs().max().item()
print("step 799: residual, initial, boundary loss", [f"{v:.1e}" for v in history[-1]], f" max error {max_err:.4f}")
```

# Checks
```python
with torch.no_grad():
    true_err = (net(xt_test) - u_exact).abs().max().item()
    mid0 = net(torch.tensor([[0.5, 0.0]])).item()
    ends = net(torch.tensor([[0.0, 0.5], [1.0, 0.5]]))[:, 0].abs().max().item()
last = history[-1] if len(history) else (float("nan"),) * 3
expect("history has one (res, ic, bc) entry per step", 0 < len(history) <= 1500 and all(len(h) == 3 for h in history), f"{len(history)} entries; want one 3-tuple per step, at most 1500 steps")
expect("u(0.5, 0) is about sin(pi/2) = 1", abs(mid0 - 1) < 0.05, f"u(0.5, 0) = {mid0:.3f}, initial loss {last[1]:.2g}." + (" Is the initial term in the loss you call backward() on?" if abs(mid0) < 0.3 else "") + (" Is the initial target shaped (16, 1)? A (16,) target against net(x0) of shape (16, 1) broadcasts to (16, 16)." if last[1] > 0.05 else ""))
expect("the ends stay at 0", ends < 0.05, f"largest |u| at x = 0 or 1 (t = 0.5) is {ends:.3f}, boundary loss {last[2]:.2g}")
expect("max error < 0.05", true_err < 0.05, f"max |net - exact| = {true_err:.3g}; last residual, initial, boundary loss: {last[0]:.2g}, {last[1]:.2g}, {last[2]:.2g}. The largest part names the condition the network missed.")
check("max_err", max_err, want=true_err, atol=1e-6)
```

# Hints
1. Each part is a mean of squares: `heat_residual(net, xt)`, `net(x0) - u0`, `net(xb)`.
2. Call `backward()` on the sum; record each part with `.item()`.

# Takeaway
Over seeds 0-9 the max error was 0.008-0.012, about 5 seconds per run.
Leaving out the initial term gave error 1.0 (measured at lr 5e-3): the
network found `u = 0`, which satisfies the equation and the ends.

Here `K` is known, which is the case a classic solver handles better. With
`K` made trainable and a data loss added, the same loop solves the inverse
problem.

What a PINN step costs, in the units of this lesson:

- Weight gradient: the derivative of the loss with respect to every
  parameter, from one backward pass of about 2 forwards of matrix-product
  FLOPs. Its memory is the forward's saved values, 1 A per hidden layer with
  tanh (2 A with sin, SiLU or GELU, which also keep their input).
- Input derivative: a backward pass of its own, toward the inputs. One pass
  gives a derivative per point only because the points do not interact.
- In a loss, the input derivative needs `create_graph=True`, which records the
  pass as new graph nodes; the weight backward then walks those too.
- Each pass stacked on the previous one doubles the matrix-product FLOPs and,
  with tanh, triples the saved tensors per layer: a second-order residual step costs 12F
  and 9 A per hidden layer, a plain step 3F and 1 A.
- Sibling passes (`2g-3`) add their costs instead of multiplying them.

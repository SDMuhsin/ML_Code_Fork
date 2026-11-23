---
title: Weighting the boundary
difficulty: Medium
devices: [cpu]
uses: [2b-2, 2d-3]
timeout: 90
inspect: [poisson_loss, W_BC, max_err]
---
# Lesson
## Why are the ends wrong when both losses are small?

Change the boundary values of the Poisson problem to `u(0) = u(1) = 1`. The
exact solution is now `1 + sin(pi x)`. So far the two parts had weight 1. A
**loss weight** is a factor on one part:

    L = residual loss + w_bc * boundary loss

The demo trains with a small and a large weight (about 4 seconds):

```python demo
torch.set_num_threads(1)
for w in (0.01, 1e4):
    torch.manual_seed(0)
    net = demo_net()
    print(f"w_bc = {w}")
    res, bc, err = train_poisson(net, w_bc=w, u_end=1.0)
    with torch.no_grad():
        ends = net(torch.tensor([[0.0], [1.0]]))[:, 0].tolist()
    print(f"  max error {err:.3f}, u(0) = {ends[0]:.3f}, u(1) = {ends[1]:.3f}  (both should be 1)")
```

The last printed step, seed 0:

| `w_bc` | residual loss | boundary loss | max error |
|---|---|---|---|
| 0.01 | 4e-5 | 0.021 | 0.15 |
| 1e4 | 8.1 | 4e-7 | 0.35 |

`w_bc` scales the boundary part's gradient, and Adam steps along the sum.
At `w_bc = 0.01` the residual is fitted and the network settles on a member
of the family `sin(pi x) + a x + b` from `2e-3` with the wrong `a` and `b`.
At `w_bc = 1e4` the boundary gradient sets the direction and, through
Adam's scaling (`2c-1`), the size of every step. The residual's share is
tiny; the steps go into two points that are already right.

## Reading the boundary loss as a miss at the ends

The boundary loss is the mean squared miss at the two ends, so
**its square root is the typical miss there**, in the units of `u`.
At `w_bc = 0.01`, `sqrt(0.021) = 0.145`, about the max error of 0.15: the
ends account for the error. At `w_bc = 1e4`,
`sqrt(4e-7) = 0.0006`, far below the error of 0.35: the ends are right
and the inside is not.

If `sqrt(boundary loss)` is more than about half your error target, raise
`w_bc`. If it is well
below the target and the max error is still too large, lower `w_bc`. Do not
lower it because the residual loss is larger than the boundary loss; one is
a squared error in `u''`, the other in `u`.

## Building the boundary into the network

A second way avoids `w_bc`. Write the output as
`u(x) = 1 + x (1 - x) net(x)`. It equals 1 at both ends for any weights, so
the boundary loss goes. This is a **hard constraint**; a loss term is a
**soft** one. It adds two elementwise operations on `(N, 1)` tensors. It needs a
formula with the boundary values (here 1) and one that is 0 on the whole
boundary and not 0 inside (here `x (1 - x)`): easy on an interval or a box,
extra work on an irregular shape, so most PINN code uses the soft loss.

```python demo
torch.manual_seed(0)
res, bc, err = train_poisson(demo_net(), u_end=1.0, hard=True)
print(f"hard constraint: max error {err:.1e}")
```

# Task
- `poisson_loss(net, w_bc)`: returns `(loss, parts)` for the problem in Given.
  `loss` is the residual loss plus `w_bc` times the boundary loss (a tensor you
  can call `backward()` on); `parts` is a dict with the two parts as Python
  floats under the keys `"res"` and `"bc"`. Take `u''` with your `ddx` twice.
- `W_BC`: a weight that gets the max error below 0.005, on the given `net`
  and on a second network with seed 5 that the checks train with
  `train_poisson`. A weight should work for more than one starting point.
- Train the given `net` for 500 steps with Adam, lr 1e-2, on
  `poisson_loss(net, W_BC)`; then `max_err`, the largest
  `|net(x) - (1 + sin(pi x))|` on `x_test`, under `torch.no_grad()`.

# Given
```python
torch.set_num_threads(1)
x = torch.linspace(0, 1, 64)[:, None].requires_grad_()   # collocation points
f = -math.pi ** 2 * torch.sin(math.pi * x.detach())      # source term
xb = torch.tensor([[0.0], [1.0]])                        # boundary points, u should be 1 there
x_test = torch.linspace(0, 1, 201)[:, None]
torch.manual_seed(0)
net = MLP()
```

# Starter
```python
def poisson_loss(net, w_bc):
    TODO

W_BC = TODO

opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for step in range(500):
    opt.zero_grad()
    # TODO: loss from poisson_loss, backward, step
max_err = TODO
```

# Solution
```python
def poisson_loss(net, w_bc):
    res_loss = ((ddx(ddx(net(x), x), x) - f) ** 2).mean()
    bc_loss = ((net(xb) - 1) ** 2).mean()
    return res_loss + w_bc * bc_loss, {"res": res_loss.item(), "bc": bc_loss.item()}

W_BC = 100.0

opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for step in range(500):
    opt.zero_grad()
    loss, parts = poisson_loss(net, W_BC)
    loss.backward()
    opt.step()
with torch.no_grad():
    max_err = (net(x_test) - 1 - torch.sin(math.pi * x_test)).abs().max().item()
print(f"step 499: residual loss {parts['res']:.2g}, boundary loss {parts['bc']:.2g}, max error {max_err:.4f}")
```

# Checks
```python
torch.manual_seed(5)
cnet = demo_net()
c_res = ((ddx(ddx(cnet(x), x), x) - f) ** 2).mean().item()
c_bc = ((cnet(xb) - 1) ** 2).mean().item()
c_out = poisson_loss(cnet, 0.01)
expect("poisson_loss returns (loss, parts)", isinstance(c_out, tuple) and len(c_out) == 2 and isinstance(c_out[1], dict), f"got {type(c_out).__name__}; return a tuple (loss, parts)")
expect("parts has the keys 'res' and 'bc'", set(c_out[1]) == {"res", "bc"}, f"keys {sorted(c_out[1])}")
check("parts['res'] (mean squared residual u'' - f)", c_out[1]["res"], want=c_res, rtol=1e-4)
check("parts['bc'] (mean squared miss of u = 1 at both ends)", c_out[1]["bc"], want=c_bc, rtol=1e-4)
for w in (0.01, 100.0):
    check(f"loss with w_bc = {w}", poisson_loss(cnet, w)[0].item(), want=c_res + w * c_bc, rtol=1e-4)
expect("the loss can be trained", poisson_loss(cnet, 1.0)[0].grad_fn is not None, "the loss has no grad_fn: return the tensor, not .item()")
with torch.no_grad():
    true_err = (net(x_test) - 1 - torch.sin(math.pi * x_test)).abs().max().item()
end_res, end_bc = poisson_loss(net, 1.0)[1].values()
def bc_advice(w, res, bc, err, base=None):
    miss = math.sqrt(max(bc, 0.0))
    msg = f"max error {err:.3g}, residual loss {res:.2g}, boundary loss {bc:.2g}, so the ends miss by about sqrt({bc:.2g}) = {miss:.4f}; the target is 0.005."
    if miss > 0.0025:
        msg += f" That is more than half the target, and the inside adds its own error: the boundary pull is too weak, raise w_bc above {w:g}."
        if w < 1:
            msg += " A weight below 1 weakens the boundary pull."
            if base is not None and miss > base:
                msg += f" The boundary got worse: at w_bc = 1 the ends missed by {base:.4f}, at w_bc = {w:g} by {miss:.4f}."
            elif base is not None:
                msg += f" At w_bc = 1 the ends missed by {base:.4f}; lowering the weight did not bring the miss under half the target."
    elif err >= 0.005:
        msg += f" The ends are within the target and the inside is not: lower w_bc below {w:g}."
    return msg
expect("max error < 0.005 on the given net", true_err < 0.005, bc_advice(float(W_BC), end_res, end_bc, true_err))
check("max_err", max_err, want=true_err, atol=1e-6)
torch.manual_seed(5)
s_res, s_bc, s_err = train_poisson(demo_net(), w_bc=float(W_BC), u_end=1.0, show=())
torch.manual_seed(5)
base_miss = math.sqrt(train_poisson(demo_net(), w_bc=1.0, u_end=1.0, show=())[1]) if (s_err >= 0.005 and float(W_BC) < 1) else None
expect(f"seed 5 with w_bc = {W_BC}: max error < 0.005", s_err < 0.005, "seed 5: " + bc_advice(float(W_BC), s_res, s_bc, s_err, base_miss))
```

# Hints
1. The boundary loss is the mean squared miss of `net(xb)` from 1.
2. Take the square root of the boundary loss your weight leaves and compare it with 0.005. Above about half of that, the weight is too small.
3. Move the weight by factors of 10 and rerun; watch `sqrt(boundary loss)` and the max error together.

# Takeaway
On seeds 0-5, `w_bc = 1` gave max errors below 0.001 on five seeds and 0.022
on seed 5; `w_bc = 100` stayed below 0.001 on all six. A boundary weight above
1 costs the residual little here, because two points are easy to fit, but `1e4` showed that it
has a ceiling. With the soft loss the weight is a setting you tune per
problem, like the learning rate, by reading the printed parts. The hard
constraint gave max errors of 1e-5 to 1e-4 on the same six seeds, with no
weight to tune.

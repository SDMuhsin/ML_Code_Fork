---
title: ReLU cannot bend
difficulty: Easy
devices: [cpu]
uses: [2b-2]
timeout: 60
inspect: [stuck_loss, ACT, fixed_err]
---
# Lesson
## Why does the loss not move at all?

Build the Poisson PINN of `2e-3` with ReLU instead of tanh. The run below is
the same training as `2e-3` (500 steps, Adam, lr 1e-2, about 2 seconds), only
the activation function differs:

```python demo
torch.set_num_threads(1)
torch.manual_seed(0)
res, bc, err = train_poisson(demo_net(act=nn.ReLU))
print(f"max error {err:.2f}")
```

The boundary loss went to 0. The residual loss is 47.94 at step 0 and at step
499, to every printed digit. The max error is 1.00: the network outputs about
0 everywhere.

A ReLU network is piecewise straight (`2b-2`). On each straight piece the
second derivative is 0, and autograd returns 0 at the kinks too. The demo
takes `u''` at 7 points with both activation functions:

```python demo
x7 = torch.linspace(0.05, 0.95, 7)[:, None].requires_grad_()
for act in (nn.ReLU, nn.Tanh):
    torch.manual_seed(0)
    net = demo_net(act=act)
    print(f"{act.__name__:5s} u'':", [round(v, 4) for v in ddx(ddx(net(x7), x7), x7)[:, 0].tolist()])
```

So for any ReLU weights the residual `u'' - f` equals `-f`, and the residual
loss equals `mean(f**2)`, a number fixed by the source term alone. Its weight
gradient is 0: no step can lower it. Only the boundary loss trains, and it
is satisfied by `u = 0`.

An equation with `u''` needs an activation function that is smooth (it has a
second derivative everywhere) and whose second derivative is not 0 over the
inputs the layers see. `tanh` has `tanh'' = -2 tanh (1 - tanh**2)`. `sin`,
SiLU (`nn.SiLU`) and Softplus also qualify; they keep more memory per layer
(`2g-2`).

# Task
- `stuck_loss`: the residual loss every ReLU network gets on this problem,
  computed from the source term `f` in Given alone (a Python float).
- `ACT`: an activation function class from `nn` whose second derivative is not 0.
- `fixed_err`: build `MLP(act=ACT)`, train it with `train_poisson` from
  Shared, and keep the max error it returns. The checks train their own copy
  and compare.

# Given
```python
torch.set_num_threads(1)
x = torch.linspace(0, 1, 64)[:, None]
f = -math.pi ** 2 * torch.sin(math.pi * x)   # the source term at the 64 collocation points
torch.manual_seed(0)
```

# Starter
```python
stuck_loss = TODO

ACT = TODO
fixed_err = TODO
```

# Solution
```python
stuck_loss = (f ** 2).mean().item()

ACT = nn.Tanh
net = MLP(act=ACT)
res, bc, fixed_err = train_poisson(net)
print(f"tanh: max error {fixed_err:.4f}")
```

# Checks
```python
xr = torch.linspace(0, 1, 64)[:, None].requires_grad_()
torch.manual_seed(1)
relu_net = demo_net(act=nn.ReLU)
relu_res = ((ddx(ddx(relu_net(xr), xr), xr) - f) ** 2).mean().item()
expect("stuck_loss is a float", isinstance(stuck_loss, float), f"stuck_loss is {type(stuck_loss).__name__}; use .item()")
expect("stuck_loss matches a ReLU network's residual loss within 1%", abs(stuck_loss - relu_res) <= 0.01 * relu_res, f"got {stuck_loss:.4g}, a ReLU network has {relu_res:.4g}. With u'' = 0 the residual is -f; what is the mean of its square?")
act_name = getattr(ACT, "__name__", str(ACT))
x7 = torch.linspace(0.05, 0.95, 7)[:, None].requires_grad_()
torch.manual_seed(0)
upp = ddx(ddx(MLP(act=ACT)(x7), x7), x7).abs().max().item()
expect("MLP(act=ACT) can bend: u'' is not 0", upp > 1e-4, f"u'' of MLP(act={act_name}) is 0 at all 7 test points: like ReLU, this activation is made of straight pieces")
torch.manual_seed(0)
own = train_poisson(MLP(act=ACT), show=())[2]
expect(f"MLP(act={act_name}) trains to max error < 0.06", own < 0.06, f"the checks' own run of MLP(act={act_name}) ended at max error {own:.3f}. Its second derivative is not 0, but it is small or flat over part of the input range; tanh's is neither.")
expect("fixed_err comes from a training run", isinstance(fixed_err, float) and 1e-7 < fixed_err < 0.06, f"fixed_err is {fixed_err!r}; want the max error train_poisson returned for your MLP(act=ACT), a float between 0 and 0.06 (the checks' run gave {own:.4f})")
```

# Hints
1. Write out the residual when `u''` is 0 everywhere.
2. `MLP` takes the activation function class, as in `2b-2`: `act=nn.Something`. Which activation function's second derivative did the lesson write out?

# Takeaway
The tanh run reached max error 0.0004 on seed 0. `nn.ELU` reaches about 0.05:
its second derivative is 0 for positive inputs. The stuck value
on the 64-point grid is 47.94; on the continuous interval it is `pi**4 / 2 =
48.7`. A residual loss that stays at the mean square of the source term
from step 0 means the residual does not depend on the weights. ReLU is one
cause. `2d-3` showed another: a derivative taken with `retain_graph` instead
of `create_graph` is a constant to the weight gradient. If `u_xx` is taken
that way, the residual `u_xx - f` again gets no weight gradient.

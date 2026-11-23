---
title: Carry the derivative along
difficulty: Medium
inspect: [jet1]
---
# Lesson
## Could the forward pass compute u' itself?

`ddx` runs after the forward pass and walks its graph backwards. Instead,
follow, for each layer's output `h`, its derivative `h'` with
respect to the input `x`, and update both as you go forward.

- At the input, `x' = dx/dx = 1`.
- A Linear layer `h = z @ W.T + b` gives `h' = z' @ W.T`. The bias is a
  constant, so it adds nothing to the derivative.
- tanh gives `t = tanh(h)` and, by the chain rule, `t' = (1 - t**2) * h'`.

So each layer maps the pair `(h, h')` to a new pair, and at the end the pair
is `(u, u')`. The demo does this by hand for a network with one hidden layer
of width 4, at 3 points, and compares with `ddx`. It uses float64
(`.double()`: numbers with about 16 significant digits instead of 7), so that
rounding cannot hide a wrong formula:

```python demo
torch.manual_seed(0)
net = demo_net(width=4, depth=1).double()
lin1, lin2 = net[0], net[2]
x = torch.tensor([[-1.0], [0.0], [0.5]], dtype=torch.float64)
h, h1 = x, torch.ones_like(x)                                  # value and derivative at the input
h, h1 = F.linear(h, lin1.weight, lin1.bias), h1 @ lin1.weight.T  # Linear: same weight, no bias on h1
t = torch.tanh(h)
h, h1 = t, (1 - t * t) * h1                                    # tanh: chain rule
h, h1 = F.linear(h, lin2.weight, lin2.bias), h1 @ lin2.weight.T
xr = x.clone().requires_grad_()
print("carried u':", h1[:, 0].tolist())
print("ddx:       ", ddx(net(xr), xr)[:, 0].detach().tolist())
```

They agree to the last digit. The derivative is exact: it is the chain rule
applied layer by layer, not a finite difference.

Pushing derivatives forward with the values is called **forward mode**, as
opposed to the backward pass of `2d-2`. The values a point carries, here
`(h, h')`, are its **jet**, and each entry of the jet is a **channel**. A
Linear layer applies the same weight to every channel and adds the bias to
the value channel only.

# Task
`jet1(net, x)`: return `(u, u_x)` for any `demo_net(din=1, ...)` on `x` of
shape `(N, 1)`, both `(N, 1)`, by carrying `(h, h')` through the layers of
`net` as in the demo. Loop over `net`; a layer is either an `nn.Linear` or an
`nn.Tanh`. Do not call `autograd.grad`, and do not mark `x` with
`requires_grad_()`.

# Starter
```python
def jet1(net, x):
    h, h1 = x, torch.ones_like(x)        # value and d/dx at the input
    for m in net:
        if isinstance(m, nn.Linear):
            TODO
        else:                            # nn.Tanh
            TODO
    return h, h1
```

# Solution
```python
def jet1(net, x):
    h, h1 = x, torch.ones_like(x)        # value and d/dx at the input
    for m in net:
        if isinstance(m, nn.Linear):
            h, h1 = F.linear(h, m.weight, m.bias), h1 @ m.weight.T
        else:                            # nn.Tanh
            t = torch.tanh(h)
            h, h1 = t, (1 - t * t) * h1
    return h, h1
```

# Checks
```python
torch.manual_seed(1)
xs = torch.linspace(-1, 1, 9, dtype=torch.float64)[:, None]
for depth in (1, 3):
    cnet = demo_net(depth=depth).double()
    cu, cux = jet1(cnet, xs)
    xr = xs.clone().requires_grad_()
    check(f"u from jet1 = net(x), depth {depth}", cu, want=cnet(xs), shape=(9, 1), atol=1e-10)
    check(f"u_x from jet1 = ddx, depth {depth}", cux, want=ddx(cnet(xr), xr), shape=(9, 1), atol=1e-10)
expect("x was not marked requires_grad", not xs.requires_grad, "jet1 marked x with requires_grad_(); the jet needs no input graph")
cnet = demo_net().double()
bad = jet1(cnet, xs)[1]
nob = xs.clone().requires_grad_()
expect("no bias on the derivative channel", (bad - ddx(cnet(nob), nob)).abs().max().item() < 1e-8, "u_x is off: is the bias added to h1? The bias is a constant; its derivative is 0.")
```

# Hints
1. Linear: `F.linear(h, m.weight, m.bias)` for the value, `h1 @ m.weight.T` for the derivative.
2. tanh: compute `t = torch.tanh(h)` first, then the derivative from `t` and the old `h1`.

# Takeaway
`u_x` came out of the forward pass, equal to `ddx` to rounding (1e-16 in
float64). No `autograd.grad` call was made. Autograd still records this
forward with respect to the weights, so `loss.backward()` works on it as on
any forward; `2k` uses that.

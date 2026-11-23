---
title: The derivative must be differentiable
difficulty: Medium
inspect: [ddx]
---
# Lesson
## Why does the second backward fail?

A training step has two halves. The **forward pass** computes `u` and the
loss and records the graph; the backward pass (`2d-2`) walks that graph back
to the leaves.

The backward pass needs some numbers from the forward pass. The derivative of
`tanh(z)` is `1 - tanh(z)**2`, so the tanh node keeps its output. Values kept
this way are called **saved values**. A backward pass frees the saved values
of the graph it walked, so their memory is returned as soon as they are used.

In a PINN loss this causes an error. `ddx(u, t)` is a backward pass through
the graph of `u`. The loss `mean((u_t + 2u)**2)` uses `u` again, and its `backward()`
needs the values `ddx` has already freed:

```python demo
lin = nn.Linear(1, 8); out = nn.Linear(8, 1)
t = torch.linspace(0, 2, 5)[:, None].requires_grad_()
u = out(torch.tanh(lin(t)))
u_t = torch.autograd.grad(u, t, torch.ones_like(u))[0]
try:
    ((u_t + 2 * u) ** 2).mean().backward()
except RuntimeError as e:
    print("error:", str(e).split(" (")[0])
```

`retain_graph=True` keeps the saved values, so the error goes away, but the
weight gradient it gives is wrong, as the next demo shows.

## Why would du/dt change when a weight changes?

Take a network with one neuron, `u = a tanh(w t + b)`. By the chain rule
`u_t = a w (1 - tanh(w t + b)**2)`. The weights are in this formula, so the
weight gradient of the loss needs `d(u_t)/dw`, the derivative of a
derivative. Autograd gets it only by differentiating the steps that computed
`u_t`, and those steps are the backward pass of `u`: a graph of a graph.

`create_graph=True` makes a backward pass record its own steps as graph
nodes. With `retain_graph` the derivative comes back with no graph: to
autograd it is a constant, and every path from the loss through `u_t` to the
weights is dropped. The demo compares the first layer's weight gradient with
finite differences (`fd_grad` in Shared nudges each weight by 1e-6):

```python demo
torch.manual_seed(0)
net = demo_net().double()
t = torch.linspace(0, 2, 64, dtype=torch.float64)[:, None].requires_grad_()
t0 = torch.zeros(1, 1, dtype=torch.float64)
W = net[0].weight
def loss_with(**flags):
    u = net(t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), **flags)[0]
    return ((u_t + 2 * u) ** 2).mean() + ((net(t0) - 1) ** 2).mean(), u_t
fd = fd_grad(lambda: loss_with(create_graph=True)[0], W)
for flags in ({"retain_graph": True}, {"create_graph": True}):
    net.zero_grad()
    loss, u_t = loss_with(**flags)
    loss.backward()
    print(flags, " u_t.grad_fn is None:", u_t.grad_fn is None, "  weight gradient relative error:", f"{rel_err(W.grad, fd):.2g}")
```

Over seeds 0-4 the `retain_graph` error was 28% to 98%, and training with it
still lowers the loss, which hides the bug. `create_graph=True` matched to
about 1e-9. It also keeps the saved values (`retain_graph` defaults to the value of
`create_graph`), so you do not need to pass both. **Whenever a loss contains an input
derivative, compute that derivative with `create_graph=True`.**

# Task
`ddx(u, x)`: your `2d-1` function with `create_graph=True` added. Every later
problem uses this version. The checks compare the weight gradient through
your `ddx` with finite differences, and print the `retain_graph` error on the
same network.

# Starter
```python
def ddx(u, x):
    TODO
```

# Solution
```python
def ddx(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
```

# Checks
```python
torch.manual_seed(1)
cnet = demo_net().double()
ct = torch.linspace(0, 2, 64, dtype=torch.float64)[:, None].requires_grad_()
ct0 = torch.zeros(1, 1, dtype=torch.float64)
CW = cnet[0].weight
def closs(deriv):
    u = cnet(ct)
    return ((deriv(u, ct) + 2 * u) ** 2).mean() + ((cnet(ct0) - 1) ** 2).mean()
def cgrad(deriv):
    cnet.zero_grad()
    closs(deriv).backward()
    return CW.grad.clone()
cfd = fd_grad(lambda: closs(ddx), CW)
expect("ddx keeps the derivative's graph", ddx(cnet(ct), ct).grad_fn is not None, "ddx(u, x).grad_fn is None: the result is a constant to autograd. Which flag records the derivative's computation?")
err_create = rel_err(cgrad(ddx), cfd)
expect("weight gradient through ddx matches finite differences", err_create < 1e-5, f"relative error {err_create:.3g}, want < 1e-5")
retain = lambda u, x: torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True)[0]
print(f"weight gradient relative error on this network: create_graph {err_create:.2g}, retain_graph {rel_err(cgrad(retain), cfd):.2g}")
check("ddx still gives 3 cos(3x)", ddx(torch.sin(3 * ct), ct), want=3 * torch.cos(3 * ct.detach()), atol=1e-6)
```

# Hints
1. One keyword argument added to the `2d-1` call.
2. If the first check says `grad_fn is None`, the flag you added keeps values but records nothing.

# Takeaway
A backward pass frees the saved values of the graph it walked.
`retain_graph=True` keeps them; it is the right flag when you need two
backward passes through one unchanged graph. A loss with `u_t` in it needs
more: its weight gradient contains `d(u_t)/dw`, which autograd gets by
differentiating the backward pass of `u`, so that pass must be recorded.
Module `2g` measures what the recorded pass costs.

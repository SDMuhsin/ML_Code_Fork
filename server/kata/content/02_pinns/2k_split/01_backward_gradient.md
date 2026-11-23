---
title: backward with a given gradient
difficulty: Medium
uses: [2j-3]
inspect: [poisson_split_step]
---
# Lesson
## backward() on a tensor that is not one number

In `2d-1`, `grad_outputs` made `autograd.grad` differentiate the weighted sum
`(grad_outputs * u).sum()`. `backward()` takes the same argument, named
`gradient`: `y.backward(gradient=G)` fills `.grad` with the gradient of
`(G * y).sum()`, with `G` shaped like `y`.

```python demo
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0])
y = w * x
try:
    y.backward()
except RuntimeError as e:
    print("y.backward():", e)
G = torch.tensor([0.5, -1.0, 4.0])
y.backward(gradient=G)
print("w.grad after y.backward(gradient=G):", w.grad.item(), "  sum(G * x):", (G * x).sum().item())
```

## Why this splits a step in two

Let `L` be a loss computed from `y`. Suppose `G` holds `dL/dy` for every
entry of `y`. By the chain rule the weight gradient of `L` is the sum over the
entries of `dL/dy` times `dy/dw`, which is the weight gradient of
`(G * y).sum()`. So `y.backward(gradient=G)` gives the weight gradient of `L`
without `L` being in the graph at all.

In a PINN, take `y` to be the **field stack**: the fields the residual reads,
side by side, `(N, n_fields)`. The derivative of the loss with respect to one
field at one point, `dL/du_xx[i]`, is called that field's **adjoint**. A step
is then:

1. jet forward, and the field stack from it (recorded by autograd);
2. residual, loss and adjoints, from the stack's values, under
   `torch.no_grad()` (no graph);
3. `stack.backward(gradient=adjoints)`: one ordinary backward pass through
   the jet.

For the 2D Poisson residual `r = u_xx + u_yy - f` with `L = mean(r**2)`, the
stack is `[u_xx, u_yy]` and both adjoints are `2 r / N`: `dL/dr[i] = 2 r[i] / N`,
and `r` grows by 1 when `u_xx[i]` or `u_yy[i]` grows by 1.

# Task
`poisson_split_step(net, xy, f)`: one split step for `net = demo_net(din=2)`
on points `xy` `(N, 2)` with source `f` `(N, 1)`. Use `jet2d` (your `2j-3`
function), build the stack `torch.cat([J[3], J[4]], 1)`, compute `r`, the
loss and the adjoint stack under `torch.no_grad()`, then call
`stack.backward(gradient=...)`. Return the loss as a Python float. The caller
zeroes `.grad` before.

# Starter
```python
def poisson_split_step(net, xy, f):
    J = jet2d(net, xy)
    stack = torch.cat([J[3], J[4]], 1)          # (N, 2): u_xx, u_yy
    with torch.no_grad():
        r = TODO
        loss = TODO
        adj = TODO                              # (N, 2), dL/du_xx and dL/du_yy
    TODO
    return loss.item()
```

# Solution
```python
def poisson_split_step(net, xy, f):
    J = jet2d(net, xy)
    stack = torch.cat([J[3], J[4]], 1)          # (N, 2): u_xx, u_yy
    with torch.no_grad():
        r = stack[:, 0:1] + stack[:, 1:2] - f
        loss = (r ** 2).mean()
        a = 2 * r / r.shape[0]
        adj = torch.cat([a, a], 1)              # (N, 2), dL/du_xx and dL/du_yy
    stack.backward(gradient=adj)
    return loss.item()
```

# Checks
```python
torch.manual_seed(4)
cnet = demo_net(din=2).double()
cxy = torch.rand(50, 2, dtype=torch.float64)
cf = torch.sin(3 * cxy[:, 0:1]) * cxy[:, 1:2]
cnet.zero_grad()
Jr = jet2d(cnet, cxy)
plain = ((Jr[3, :, 0:1] + Jr[4, :, 0:1] - cf) ** 2).mean()
plain.backward()
g_plain = weight_grads(cnet).clone()
cnet.zero_grad()
val = poisson_split_step(cnet, cxy, cf)
g_split = weight_grads(cnet)
expect("returns the loss as a Python float", isinstance(val, float), f"got {type(val).__name__}")
check("loss value", val, want=plain.item(), atol=1e-10)
err = rel_err(g_split, g_plain)
expect("weight gradients = plain loss.backward()", err < 1e-8, f"relative error {err:.3g}. Both adjoints are 2 r / N; did you divide by N and multiply by 2?")
cnet.zero_grad()
n_split = count_saved(lambda: poisson_split_step(cnet, cxy, cf))[0]
n_jet = count_saved(lambda: jet2d(cnet, cxy))[0]
expect("no graph for the residual", n_split == n_jet, f"the step saved {n_split} tensors, the jet alone {n_jet}: compute r, the loss and adj under torch.no_grad()")
```

# Hints
1. `r` uses the two columns of `stack`, kept `(N, 1)` with `0:1` and `1:2`.
2. `adj` puts the same `(N, 1)` tensor `2 * r / N` in both columns.

# Takeaway
The split step gave the same weight gradients as `loss.backward()` (to 1e-16
in float64), and the residual and loss were computed with no graph. The
second half is one ordinary backward pass through the jet. The first half,
the adjoints, was written by hand. For `u_xx + u_yy - f` that was one line; the next problem
does it for a residual with products.

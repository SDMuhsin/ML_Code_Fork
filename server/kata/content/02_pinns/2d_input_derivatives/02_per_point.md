---
title: Why ones gives per-point derivatives
difficulty: Medium
uses: [2b-2, 2d-1]
inspect: [per_point_ddx, gap_mlp, gap_coupled]
---
# Lesson
## One call, N derivatives: when is that allowed?

`ddx` differentiates the sum `u[0] + ... + u[N-1]` and gets back N numbers.
To see what those numbers are, look at all the derivatives at once. The
**Jacobian** of `u` (N values) with respect to `x` (N values) is the N x N
table whose entry `(i, j)` is `du[i]/dx[j]`: how output `i` changes when
input `j` moves. Entry `j` of the sum's derivative is the sum of column `j`
of that table.

You want the diagonal, `du[j]/dx[j]`. A column sum equals the diagonal entry
exactly when the rest of the column is 0, that is, when output `i` does not
depend on input `j` for `i != j`. An MLP applied to `x` of shape `(N, 1)`
computes row `i` of the output from row `i` of the input only. The demo
prints the Jacobian of an MLP on 4 points (`torch.autograd.functional.jacobian`
builds the whole table; it is used here to look, not in training):

```python demo
torch.manual_seed(0)
net = demo_net()
x = torch.linspace(-1, 1, 4)[:, None]
J = torch.autograd.functional.jacobian(lambda z: net(z)[:, 0], x)[:, :, 0]
print(J)
print("largest off-diagonal entry:", (J - torch.diag(torch.diag(J))).abs().max().item())
```

Now a function that first subtracts the batch mean, the average of all N
inputs: `u = sin(3 (x - mean(x)))`. Every output now depends on every input.

```python demo
coupled = lambda z: torch.sin(3 * (z - z.mean()))
Jc = torch.autograd.functional.jacobian(lambda z: coupled(z)[:, 0], x)[:, :, 0]
print(Jc)
print("diagonal:   ", [round(v, 4) for v in torch.diag(Jc).tolist()])
print("column sums:", [round(v, 4) for v in Jc.sum(0).tolist()])
```

The column sums are what `grad_outputs = ones` returns, and they are not the
per-point derivatives. The sum trick is correct only because the points do
not interact. Anything that mixes rows of the input breaks it without an
error.

The walk that `backward()` or `torch.autograd.grad` makes from one result
back to the leaves is called a **backward pass**. A backward pass always
differentiates one number, and in one walk it gives that number's derivative
with respect to every input. That is how one `L.backward()` in `2c` filled
the gradients of all 2209 parameters: one loss, many weights.

N outputs need N backward passes, one per output, unless the Jacobian is
diagonal; then one pass with `grad_outputs = ones` gives all N.

# Task
- `per_point_ddx(f, x)`: the per-point derivative computed one point at a time. For
  each `i`, make a fresh tracked copy of `x` (`x.clone().requires_grad_()`),
  compute `f` of it, and take the derivative of output `i` alone with
  respect to that copy; keep entry `i`. Return shape `(N, 1)`. It calls
  `torch.autograd.grad` N times.
- `gap_mlp`, `gap_coupled`: the largest absolute difference (a Python float)
  between `per_point_ddx` and the sum trick `ddx` (your `2d-1` function) for
  the given `net` and for `coupled`, both on the given `x`. The given `x`
  is not tracked: pass `ddx` a tracked copy (`xr = x.clone().requires_grad_()`)
  and the output computed from that copy, or it raises "does not require grad".

# Given
```python
torch.manual_seed(0)
net = MLP()
x = torch.linspace(-1, 1, 6)[:, None]
coupled = lambda z: torch.sin(3 * (z - z.mean()))
```

# Starter
```python
def per_point_ddx(f, x):
    TODO

gap_mlp = TODO
gap_coupled = TODO
```

# Solution
```python
def per_point_ddx(f, x):
    out = []
    for i in range(x.shape[0]):
        xi = x.clone().requires_grad_()
        ui = f(xi)[i, 0]
        out.append(torch.autograd.grad(ui, xi)[0][i, 0])
    return torch.stack(out)[:, None]

def sum_trick(f, x):
    xr = x.clone().requires_grad_()
    return ddx(f(xr), xr)

gap_mlp = (per_point_ddx(net, x) - sum_trick(net, x)).abs().max().item()
gap_coupled = (per_point_ddx(coupled, x) - sum_trick(coupled, x)).abs().max().item()
```

# Checks
```python
diag = lambda f: torch.diag(torch.autograd.functional.jacobian(lambda z: f(z)[:, 0], x)[:, :, 0])[:, None]
check("per_point_ddx(net, x) = Jacobian diagonal", per_point_ddx(net, x), want=diag(net), shape=(6, 1), atol=1e-6)
xs_ = x.clone().requires_grad_()
expect("per_point_ddx(coupled) differs from the sum trick", (per_point_ddx(coupled, x) - ddx(coupled(xs_), xs_)).abs().max().item() > 0.1, "your loop returns column sums: it differentiates the sum of all outputs. Take output i alone: f(xi)[i, 0].")
check("per_point_ddx(coupled, x) = Jacobian diagonal", per_point_ddx(coupled, x), want=diag(coupled), shape=(6, 1), atol=1e-5)
xr = x.clone().requires_grad_()
true_gap = (diag(coupled) - ddx(coupled(xr), xr)).abs().max().item()
expect("gap_mlp is about 0", isinstance(gap_mlp, float) and gap_mlp < 1e-6, f"gap_mlp = {gap_mlp}; want a Python float near 0 (use .item())")
check("gap_coupled", gap_coupled, want=true_gap, atol=1e-5)
expect("the sum trick fails on coupled", gap_coupled > 0.1, f"gap_coupled = {gap_coupled}; did you compare against the sum trick on coupled, not on net?")
```

# Hints
1. Inside the loop, `f(xi)[i, 0]` is one number, so `torch.autograd.grad` needs no `grad_outputs` for it.
2. For the sum trick, `ddx` needs a tracked `x` and `u` computed from it: clone, `requires_grad_()`, then call `f`.

# Takeaway
On the MLP the loop and the sum trick agreed to below 1e-6, and the loop
cost N calls instead of one. On `coupled` the sum trick was off by more than
0.1 with no error. Every PINN in this lesson uses an MLP on independent
points, so `grad_outputs = ones` is correct there. A model that mixes
points, for example through a batch mean, needs the loop. The common case is
`nn.BatchNorm1d` in training mode, which subtracts the batch mean; do not put
it in a PINN.

---
title: Adjoints by hand
difficulty: Medium
inspect: [burgers_adjoint]
---
# Lesson
## One rule per operation

Step 3 of `2k-1` computed, for each weight `w`, the sum over the stack of
`G * d(stack)/dw`. In matrix terms that is the vector `G` times a Jacobian
(`2d-2`), here of the stack with respect to the weights: a
**vector-Jacobian product (VJP)**. The same product exists for one operation
and its inputs: given the adjoint of the result, it gives the adjoints of the
inputs. A backward pass is a chain of these, one per operation, and each
operation has a short rule. With `g` the adjoint of the result `c`:

| operation | adjoint of `a` | adjoint of `b` |
|---|---|---|
| `c = a + b` | `g` | `g` |
| `c = a - b` | `g` | `-g` |
| `c = a * b` | `g * b` | `g * a` |
| `c = k * a`, `k` a constant | `k * g` | |

Walk the operations from the last to the first and apply the rule of each.
When a variable is read by two operations, or twice by one, it gets one
contribution from each, and they add, as `.grad` adds over `backward()`
calls (`2a-2`). The demo checks this on `q = u * u + u_x`: `u` gets `g * u`
twice from the product.

```python demo
u = torch.tensor([[0.5], [-1.0], [2.0]], dtype=torch.float64, requires_grad=True)
u_x = torch.tensor([[1.0], [3.0], [-2.0]], dtype=torch.float64, requires_grad=True)
q = u * u + u_x
g_q = torch.ones_like(q)                    # adjoint of q, as if L = q.sum()
p = u * u                                   # q = p + u_x
g_p, g_ux = g_q, g_q                        # add rule
g_u = g_p * u + g_p * u                     # mul rule, u read twice: two contributions
q.sum().backward()
print("by the rules: adj u =", g_u[:, 0].tolist(), " adj u_x =", g_ux[:, 0].tolist())
print("autograd:     adj u =", u.grad[:, 0].tolist(), " adj u_x =", u_x.grad[:, 0].tolist())
```

## A residual with a product

The steady 1D viscous **Burgers equation** has the residual
`r = u * u_x - nu * u_xx - f`: the flow carries `u` with speed `u`, and
viscosity `nu` smooths it. Write the loss per point as `l = r * r`, so that
`L = mean(l)` and the adjoint of every `l[i]` is `1/N`. The operations, in
order:

    p = u * u_x
    q = nu * u_xx
    s = p - q
    r = s - f
    l = r * r

# Task
`burgers_adjoint(u, u_x, u_xx, f, nu)`: the adjoints of `u`, `u_x`, `u_xx`
for `L = mean((u * u_x - nu * u_xx - f)**2)`, as one `(N, 3)` tensor
`torch.cat([adj_u, adj_u_x, adj_u_xx], 1)`. All inputs are `(N, 1)`, `nu` is
a float. Walk the five operations backwards from `adj_l = 1/N` with the rules
above. No autograd.

# Starter
```python
def burgers_adjoint(u, u_x, u_xx, f, nu):
    N = u.shape[0]
    r = u * u_x - nu * u_xx - f
    adj_l = 1.0 / N
    adj_r = TODO
    adj_s = TODO
    adj_q = TODO
    adj_p = TODO
    adj_u = TODO
    adj_u_x = TODO
    adj_u_xx = TODO
    return torch.cat([adj_u, adj_u_x, adj_u_xx], 1)
```

# Solution
```python
def burgers_adjoint(u, u_x, u_xx, f, nu):
    N = u.shape[0]
    r = u * u_x - nu * u_xx - f
    adj_l = 1.0 / N
    adj_r = adj_l * r + adj_l * r          # l = r * r: two contributions
    adj_s = adj_r
    adj_q = -adj_s
    adj_p = adj_s
    adj_u = adj_p * u_x
    adj_u_x = adj_p * u
    adj_u_xx = nu * adj_q
    return torch.cat([adj_u, adj_u_x, adj_u_xx], 1)
```

# Checks
```python
torch.manual_seed(5)
fs = [torch.randn(40, 1, dtype=torch.float64, requires_grad=True) for _ in range(3)]
cf = torch.randn(40, 1, dtype=torch.float64)
cnu = 0.07
((fs[0] * fs[1] - cnu * fs[2] - cf) ** 2).mean().backward()
want = torch.cat([t.grad for t in fs], 1)
got = burgers_adjoint(*[t.detach() for t in fs], cf, cnu)
same("shape (N, 3)", S(got), (40, 3))
ratio = (got[:, 0] / want[:, 0]).mean().item()
expect("both contributions of l = r * r", abs(ratio - 0.5) > 1e-6, "your adjoints are half the right size: l = r * r reads r twice, so adj_r = 2 r / N")
check("adj u", got[:, 0:1], want=want[:, 0:1], atol=1e-12)
check("adj u_x", got[:, 1:2], want=want[:, 1:2], atol=1e-12)
check("adj u_xx", got[:, 2:3], want=want[:, 2:3], atol=1e-12)
```

# Hints
1. Through `r = s - f`, `s` gets `adj_r`. Through `s = p - q`, `q` gets `-adj_s`.
2. Through `p = u * u_x`, `u` gets `adj_p * u_x` and `u_x` gets `adj_p * u`.

# Takeaway
Five operations took five rules, and the adjoints matched autograd to 1e-12.
The residual of `2m`, 2D Navier-Stokes, has 20 operations and its adjoint is
40 lines; the residual in `pinn-solver-accel` has 171 operations and 331
lines (with friction off; with friction the residual also uses `sqrt`, whose rule `2m-2` gives). The next module writes a program that writes these lines.

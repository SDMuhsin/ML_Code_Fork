---
title: SwiGLU
difficulty: Medium
---
# Task
Write `swiglu_mlp(x, Wg, Wu, Wd)` for `x` of shape `(..., D)`, returning `(..., D)`.

```text
act = silu(gate) * up
```

`silu` goes on the gate. `W_gate` and `W_up` are `(I, D)`; `W_down` is
`(D, I)`.

# Given
```python
D, I = 8, 16
Wg, Wu = torch.randn(I, D), torch.randn(I, D)
Wd = torch.randn(D, I)
x = torch.randn(2, 3, D)
```

# Starter
```python
def swiglu_mlp(x, Wg, Wu, Wd):
    return TODO
```

# Solution
```python
def swiglu_mlp(x, Wg, Wu, Wd):
    return F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)
```

# Checks
```python
out = swiglu_mlp(x, Wg, Wu, Wd)
check("shape", out, shape=(2, 3, D))
check("values", out, want=(F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T, atol=1e-4)
expect("silu is on the gate only",
       not torch.allclose(swiglu_mlp(x, Wg, Wu, Wd), swiglu_mlp(x, Wu, Wg, Wd), atol=1e-4),
       "swapping Wg and Wu must change the answer")
```

# Hints
1. 1h "F.linear transposes for you": all three weights are stored `(out, in)`, the layout `F.linear` expects.
2. Project `x` through the gate and the up weights, apply `F.silu` to the gate projection only, multiply elementwise, then project down with `Wd`.

# Takeaway
`silu` goes on the gate. Put it on `up` instead and everything still runs with
the right shape; only the numbers are wrong, which is why the last check swaps
the two weights.

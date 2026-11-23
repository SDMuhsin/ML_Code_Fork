---
title: "E. The MLP"
difficulty: Easy
---
# Lesson
## E. The MLP

`mlp(h, lw) -> (1, 1, D)`, with `h` already normalised.

# Task
Write `mlp(h, lw)`: the SwiGLU MLP. Project `h` through the gate `lw["wg"]`
and the up projection `lw["wu"]`, apply SiLU to the gate, multiply the two,
and project down through `lw["wd"]`. Returns `(1, 1, D)`.

# Given
```python
hx, lw0 = torch.randn(1, 1, CFG["D"]), W["layers"][0]
```

# Starter
```python
def mlp(h, lw):
    return TODO
```

# Solution
```python
def mlp(h, lw):
    return F.linear(F.silu(F.linear(h, lw["wg"])) * F.linear(h, lw["wu"]), lw["wd"])
```

# Checks
```python
ob = mlp(hx, lw0)
check("shape", ob, shape=(1, 1, CFG["D"]))
check("values", ob,
      want=F.linear(F.silu(F.linear(hx, lw0["wg"])) * F.linear(hx, lw0["wu"]), lw0["wd"]))
expect("silu is on the gate, not on up", not torch.allclose(
    ob, F.linear(F.silu(F.linear(hx, lw0["wu"])) * F.linear(hx, lw0["wg"]), lw0["wd"])),
    "your output equals silu(up) * gate: silu must be on the gate, not on up")
```

# Hints
1. Three `F.linear` calls: two going up to `I`, one coming back down to `D`. `F.silu` is the activation.
2. SiLU goes on the gate projection only; the up projection is multiplied in raw, elementwise, and that product is what `wd` projects down.

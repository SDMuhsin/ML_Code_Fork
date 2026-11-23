---
title: The verdict
difficulty: Medium
---
# Task
`report(got, ref)` giving `max_abs` and `argmax_agree` in fp64 (as Python
floats), then `verdict(got, ref)`: True iff `max_abs <= 0.5` and
`argmax_agree >= 0.985`.

# Given
```python
V = 10
ref = torch.randn(1000, V)
near = ref + torch.randn(1000, V) * 1e-3
far = torch.randn(1000, V)
```

# Starter
```python
def report(got, ref):
    g, r = got.double(), ref.double()
    return dict(max_abs=TODO, argmax_agree=TODO)


def verdict(got, ref):
    m = report(got, ref)
    return TODO
```

# Solution
```python
def report(got, ref):
    g, r = got.double(), ref.double()
    return dict(max_abs=float((g - r).abs().max()),
                argmax_agree=float((g.argmax(-1) == r.argmax(-1)).double().mean()))


def verdict(got, ref):
    m = report(got, ref)
    return m["max_abs"] <= 0.5 and m["argmax_agree"] >= 0.985
```

# Checks
```python
check("max_abs", torch.tensor(report(near, ref)["max_abs"]),
      want=(near.double() - ref.double()).abs().max())
check("verdict passes a near match", verdict(near, ref), want=True)
check("verdict fails a bad one", verdict(far, ref), want=False)
check("identical tensors pass", verdict(ref, ref), want=True)
# Each gate has to be able to fail on its own, so here is a case that trips
# only the argmax gate and one that trips only the magnitude gate.
tied = ref.clone()
tied[:, 0] = tied[:, 1]                       # near-ties, tiny max_abs
tied[:500, 0] += 1e-6                         # flips argmax on half the rows
check("a tiny perturbation that flips argmax fails", verdict(tied, ref), want=False)
shifted = ref + 0.9                           # argmax unchanged, max_abs large
check("a large uniform shift fails", verdict(shifted, ref), want=False)
check("argmax is unchanged by a uniform shift",
      torch.tensor(report(shifted, ref)["argmax_agree"]), want=1.0)
```

# Hints
1. 1l Drill 1 (`compare`) computes exactly these two numbers, in fp64.
2. `argmax_agree` is the mean of a boolean "row argmaxes match" tensor, made a Python float; `max_abs` is the largest absolute difference. `verdict` needs both gates.

# Takeaway
The bar is 98.5% argmax agreement, not bit-equality, and it comes with a
magnitude gate. Each gate has to be able to fail on its own: a near-tie can
flip argmax with a tiny `max_abs`, and a uniform shift leaves argmax intact
with a large one.

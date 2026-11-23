---
title: Exactly these ranks
difficulty: Medium
inspect: [best, best_col, top, peak]
---
# Lesson
## Rank 0

A full reduction gives a rank-0 tensor of shape `()`. It is neither a Python
float nor a length-1 vector.

```python demo
t = torch.randn(4)
s = t.sum()

print("t.sum()       ", S(s), " rank", s.ndim)
print("t.sum().item()", type(s.item()).__name__)
print("t.sum()[None] ", S(s[None]), " rank", s[None].ndim)
```

All three hold the same number. Only `.item()` gives a plain Python number,
which is what f-strings and non-torch code expect.

## amax, max and keepdim

`amax(dim)` gives the maximum values. `max(dim)` gives a `(values,
indices)` pair, which is not a tensor and will not pass a shape check.

`keepdim=True` leaves the collapsed axis behind with length 1, so the result
still lines up against the input. Module 1g covers it in full.

```python demo
t = torch.randn(2, 5)
print("t.amax(-1)              ", S(t.amax(-1)))
print("t.amax(-1, keepdim=True)", S(t.amax(-1, keepdim=True)))
print("t.max(-1)               ", type(t.max(-1)).__name__, "  not a tensor")
```

# Task
From `logits` of shape `(T, V)`, produce exactly these:

- `best`: the winning index per row, shape `(T,)`
- `best_col`: the same thing as a column, shape `(T, 1)`
- `top`: each row's largest value, shape `(T, 1)`
- `peak`: the single largest value anywhere, as a Python float

# Given
```python
T, V = 3, 10
logits = torch.randn(T, V)
```

# Starter
```python
best = TODO
best_col = TODO
top = TODO
peak = TODO
```

# Solution
```python
best = logits.argmax(-1)
best_col = logits.argmax(-1, keepdim=True)
top = logits.amax(-1, keepdim=True)
peak = logits.max().item()
```

# Checks
```python
check("best", best, shape=(T,))
check("best_col", best_col, shape=(T, 1))
check("best and best_col agree", best_col[:, 0], want=best)
check("top", top, shape=(T, 1))
check("top lines up with logits", (logits - top).amax(-1), want=torch.zeros(T))
expect("peak is a python float", isinstance(peak, float), f"peak is {type(peak).__name__}, want float. Which method turns a rank-0 tensor into a Python number?")
check("peak is the largest value", peak, want=logits.max())
```

# Hints
1. `argmax` and `amax` both take a `dim` and a `keepdim`. Which one gives indices, which one values?
2. A full reduction with no `dim` gives a rank-0 tensor; `.item()` turns that into a Python number.

# Takeaway
`top` has shape `(T, 1)` so that `logits - top` works: the kept length-1 axis
broadcasts back across the row. Without `keepdim`, `(T,)` would line up against
the last axis instead, which is `V`, and that is either an error or, on a
square tensor, silently wrong.

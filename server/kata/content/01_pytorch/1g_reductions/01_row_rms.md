---
title: Root mean square per row
difficulty: Easy
---
# Lesson
## dim says which axis to collapse

```python demo
x = torch.arange(12).float().reshape(3, 4)

print("x.sum()      ", S(x.sum()), "  everything -> rank 0")
print("x.sum(dim=0) ", S(x.sum(0)), " rows collapsed")
print("x.sum(dim=1) ", S(x.sum(1)), " columns collapsed")
```

`keepdim=True` leaves the axis behind as a length-1 blank axis.

```python demo
print("x.sum(-1)              ", S(x.sum(-1)))
print("x.sum(-1, keepdim=True)", S(x.sum(-1, keepdim=True)))
```

`keepdim` exists so that the result broadcasts back against the input. Every
normalisation you write needs it.

```python demo
sq = torch.arange(16).float().reshape(4, 4) + 1

print("with keepdim, row sums:", (sq / sq.sum(-1, keepdim=True)).sum(-1).tolist())
print("without:               ",
      [round(f, 3) for f in (sq / sq.sum(-1)).sum(-1).tolist()])
```

The second line is wrong and raised nothing, because the tensor is square.
A rectangular tensor would have raised an error.

## max and amax differ

`max(dim=...)` returns values and indices. `amax` returns only values.

```python demo
m = x.max(dim=-1)
print("x.max(-1) ->", type(m).__name__, m.values.tolist(), m.indices.tolist())
print("x.amax(-1) ->", x.amax(-1).tolist())
```

# Task
Compute `rms`: the root mean square of each row of `x` over the last axis,
shaped so that `x / rms` is correct.

# Given
```python
D = 8
x = torch.randn(3, D)
```

# Starter
```python
rms = TODO
```

# Solution
```python
rms = x.pow(2).mean(-1, keepdim=True).sqrt()
```

# Checks
```python
check("rms", rms, shape=(3, 1))
check("x / rms has unit rows", (x / rms).pow(2).mean(-1), want=torch.ones(3))
```

# Hints
1. Root mean square, read right to left: square, take the mean over the last axis, square root.
2. For `x / rms` to divide each row by its own number, `rms` must be `(3, 1)`, not `(3,)`. Which argument keeps the collapsed axis?

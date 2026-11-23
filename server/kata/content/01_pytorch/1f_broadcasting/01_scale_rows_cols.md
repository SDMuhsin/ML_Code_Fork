---
title: Scale columns, scale rows
difficulty: Easy
---
# Lesson
## The rule

PyTorch right-aligns the two shapes, then compares axis by axis from the
right:

- equal lengths: fine
- one of them is 1: that side is repeated
- a missing axis: treated as 1
- anything else: error

```python demo
for a, b in [((1, 4, 1, 2), (1, 1, 2)), ((4, 1), (1, 3)), ((2, 3), (3,))]:
    print(f"{str(a):>14} + {str(b):>10} -> {S(torch.zeros(a) + torch.zeros(b))}")
```

```python demo
try:
    torch.zeros(4, 3) + torch.zeros(4)
except RuntimeError as e:
    print("(4,3) + (4,) ->", str(e).split("at non-singleton")[0].strip())
```

`(4,)` right-aligns onto the last axis, which has length 3, not onto the first.

## The silent case

Because a missing axis is treated as 1, a misaligned operand on a square
tensor gives no error and no warning.

```python demo
ones = torch.ones(4, 4)
col = torch.tensor([1.0, 2.0, 3.0, 4.0])

print("(4,4) * (4,)        col 0 =", (ones * col)[:, 0].tolist(), " row-scaled")
print("(4,4) * (4,)[:,None] col 0 =", (ones * col[:, None])[:, 0].tolist(), " col-scaled")
```

Both results are `(4, 4)`, and only one is what you meant.

Write the blank axis explicitly instead of relying on alignment. Rectangular
shapes help too: with sizes like `4096`, `14336` and `32000`, a misaligned
operand usually raises.

# Task
`w` is an `(I, D)` weight matrix. `col_scale` has one value per input
column; `row_scale` has one per output row. Apply each:

- `col_scaled`: `w` with every column `j` multiplied by `col_scale[j]`
- `row_scaled`: `w` with every row `i` multiplied by `row_scale[i]`

Write the blank axis explicitly in both.

# Given
```python
I, D = 16, 8
w = torch.ones(I, D)
col_scale = torch.arange(1, D + 1).float()
row_scale = torch.arange(1, I + 1).float()
```

# Starter
```python
col_scaled = TODO
row_scaled = TODO
```

# Solution
```python
col_scaled = w * col_scale[None, :]
row_scaled = w * row_scale[:, None]
```

# Checks
```python
check("col_scaled row 0", col_scaled[0], want=col_scale)
check("row_scaled col 0", row_scaled[:, 0], want=row_scale)
```

# Hints
1. Each scale must line up with one axis of `w`. Which axis does a bare `(D,)` or `(I,)` line up with?
2. Index with `None` to add a length-1 axis: the scale's values go along the axis they belong to, and the `None` goes on the other one.

# Takeaway
The packed weight format in 1j carries a per-input-column scale like
`col_scale`.

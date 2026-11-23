---
title: Fix each line
difficulty: Easy
---
# Task
Each line does not do what its comment says. Fix `fix1`, `fix2` and `fix3`
so that each does.

# Given
```python
I, D, V = 8, 8, 3
w = torch.randn(I, D)
scale = torch.randn(I)
logits = torch.randn(3, V)
scores = torch.randn(1, 4, 1, 6)
bias = torch.randn(6)
```

# Starter
```python
fix1 = w * scale                    # scale row r of w by scale[r]
fix2 = logits - logits.amax(-1)     # subtract each row's own max
fix3 = scores + bias[:, None]       # add bias across the KEY axis
```

# Solution
```python
fix1 = w * scale[:, None]
fix2 = logits - logits.amax(-1, keepdim=True)
fix3 = scores + bias[None, None, None, :]
```

# Checks
```python
check("fix1 row 5", fix1[5], want=w[5] * scale[5])
check("fix2 row maxima are 0", fix2.amax(-1), want=torch.zeros(3))
check("fix3", fix3, shape=(1, 4, 1, 6))
check("fix3 key 4", fix3[0, 1, 0, 4], want=scores[0, 1, 0, 4] + bias[4])
```

# Hints
1. 1f "The silent case": a bare `(N,)` vector lines up against the last axis. 1g: a reduction without `keepdim` drops the axis it reduced.
2. `scale` and the row maxima each need a blank axis after them (`[:, None]` or `keepdim=True`). `scores` is `(1, H, Tq, Skv)`: the key axis is the last one, so `bias` needs its blanks in front.

# Takeaway
`w` and `logits` are square here, so the first two wrong lines run and give
wrong numbers with no error: omitting `keepdim` is silent on a square tensor
and an error on a rectangular one. Write the blank axis explicitly rather than
relying on alignment.

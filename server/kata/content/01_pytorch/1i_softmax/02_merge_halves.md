---
title: Merge two halves
difficulty: Medium
---
# Task
One score row is split in two. Each half is summarised as `(m, l, e)`: its
max, its `sum(exp(s - m))`, and its unnormalised `exp(s - m)` vector. You are
given `m1, l1, e1` and `m2, l2, e2`.

Then `s` is deleted, so you cannot look at it again.

Merge the two summaries into the true softmax of the whole row, `p_merged`,
shape `(10,)`, computing `m`, `c1`, `c2` and `l` on the way:

```text
m  = max(m1, m2)
c1 = exp(m1 - m)      c2 = exp(m2 - m)
l  = l1*c1 + l2*c2
p  = cat(e1*c1, e2*c2) / l
```

# Given
```python
s = torch.randn(10) * 20

half = 5
m1 = s[:half].max()
e1 = (s[:half] - m1).exp()
l1 = e1.sum()
m2 = s[half:].max()
e2 = (s[half:] - m2).exp()
l2 = e2.sum()
del s                                # enforced
```

# Starter
```python
m = TODO
c1 = TODO
c2 = TODO
l = TODO
p_merged = TODO
```

# Solution
```python
m = torch.maximum(m1, m2)
c1 = (m1 - m).exp()
c2 = (m2 - m).exp()
l = l1 * c1 + l2 * c2
p_merged = torch.cat([e1 * c1, e2 * c2]) / l
```

# Checks
```python
# For the check only: rebuild the row from the summaries, since s is gone.
s_back = torch.cat([m1 + e1.log(), m2 + e2.log()])

check("m is the max of the whole row", m, want=s_back.max())
check("l is sum(exp(s - m)) over the whole row", l, want=(s_back - s_back.max()).exp().sum())
check("shape", p_merged, shape=(10,))
check("equals the whole-row softmax", p_merged, want=F.softmax(s_back, dim=-1), atol=1e-6)
```

# Hints
1. Each half's `e` and `l` were computed against its own max. To combine them, rescale both onto the common max `m`.
2. `torch.maximum` takes the larger of two tensors. Each correction factor is an `exp` of a difference that is at most 0; apply one to each half, and divide by `l` once, at the end.

# Takeaway
Nothing after the split touched `s`.

Two details matter later and are easy to get wrong. Both halves get a
correction factor, and you divide by `l` once, at the end.

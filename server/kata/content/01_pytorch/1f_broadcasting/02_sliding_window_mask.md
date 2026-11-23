---
title: Sliding-window mask
difficulty: Medium
---
# Task
BioMistral attends over a sliding window of 4096. A query at position `p` may
see key position `j` when

```text
    j <= p          no looking ahead, itself included
and j >  p - W      nothing older than W-1 steps back
```

Build `allowed` of shape `(Tq, Skv)` (here `(3, 8)`, dtype `bool`) from
`q_pos` and `kv_pos`, using blank axes and comparisons. No loops. Watch the
inclusive and exclusive ends.

# Given
```python
W = 4
q_pos = torch.tensor([4, 5, 6])
kv_pos = torch.arange(8)
```

# Starter
```python
allowed = TODO
```

# Solution
```python
allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - W)
```

# Checks
```python
check("shape", allowed, shape=(3, 8))
check("dtype is bool", allowed, dtype=torch.bool)
check("values", allowed.long(), want=torch.tensor([
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0]]))
```

# Hints
1. Rows are queries and columns are keys. Give each position vector a blank axis so that a comparison between them broadcasts to `(3, 8)`.
2. `q_pos[:, None]` is a column, `kv_pos[None, :]` a row. Build one bool mask per condition and combine them with `&`, not `and`.

# Takeaway
Swapping the two `None`s gives the transpose. That bug is invisible when
`Tq == 1`, which is every decode step, and shows up only in prefill. Module 1l
has you find it.

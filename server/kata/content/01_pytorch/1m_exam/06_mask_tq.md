---
title: Mask at Tq > 1
difficulty: Medium
---
# Task
Window `W`. Query at `p` sees key at `j` iff `j <= p` and `j > p - W`.
Build `allowed` `(Tq, Skv)`, then `probs` `(1, H, Tq, Skv)`: `raw` masked,
softmax over the key axis in fp32.

# Given
```python
H, W = 4, 3
q_pos = torch.tensor([3, 4, 5])
kv_pos = torch.arange(7)
raw = torch.randn(1, H, 3, 7)
```

# Starter
```python
allowed = TODO
probs = TODO
```

# Solution
```python
allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - W)
probs = F.softmax(raw.masked_fill(~allowed[None, None], float("-inf")), -1,
                  dtype=torch.float32)
```

# Checks
```python
check("allowed", allowed.long(), want=torch.tensor([
    [0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0]]))
check("rows sum to 1", probs.sum(-1), want=torch.ones(1, H, 3))
check("zero where blocked", probs[0, 0, 0, 4:], want=torch.zeros(3))
```

# Hints
1. 1f Drill 2 builds this mask from blank axes and comparisons; 1i Drill 1 masks and softmaxes in fp32.
2. Queries index the rows, so `q_pos` gets the trailing blank axis and `kv_pos` the leading one. `allowed` then needs two more blank axes in front to meet `raw`; `masked_fill` writes where its mask is True, so pass `~allowed`.

# Takeaway
Swap the two `None`s and you get the transpose. That bug is invisible at
decode (`Tq == 1`) and only breaks in prefill, which is why this one is
checked at `Tq = 3`.

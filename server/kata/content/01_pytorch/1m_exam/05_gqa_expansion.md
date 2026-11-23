---
title: GQA expansion
difficulty: Medium
---
# Task
`k` is `(B, HKV, Tk, DH)`. Make `k_rep`, shape `(B, HKV*G, Tk, DH)`, with kv
head `j` serving query heads `jG .. jG+G-1`, copying at most once.

Then build `k_bad` the way that gives the wrong mapping (same shape), and
confirm it differs.

# Given
```python
B, HKV, G, Tk, DH = 2, 2, 2, 4, 2
k = torch.randn(B, HKV, Tk, DH)
```

# Starter
```python
k_rep = TODO
k_bad = TODO
```

# Solution
```python
k_rep = k[:, :, None].expand(B, HKV, G, Tk, DH).reshape(B, HKV * G, Tk, DH)
k_bad = k.repeat(1, G, 1, 1)
```

# Checks
```python
check("shape", k_rep, shape=(B, HKV * G, Tk, DH))
for j in range(HKV):
    for r in range(G):
        check(f"kv {j} -> q {j*G+r}", k_rep[:, j * G + r], want=k[:, j])
expect("the wrong way really is different", not torch.equal(k_rep, k_bad), "k_bad should differ")
```

# Hints
1. 1e Drill 2 (`repeat_kv`): two operations, and the first one copies nothing.
2. Insert a blank axis after the head axis, `expand` it to `G`, then collapse the two head axes. For `k_bad`, 1e's "expand, repeat, repeat_interleave" shows which call gives `kv0, kv1, kv0, kv1`.

# Takeaway
GQA expansion is a block repeat: `expand` then `reshape`, or
`repeat_interleave`. Never `repeat`. A `repeat`-based answer has the right
shape and the wrong contents, and nothing crashes.

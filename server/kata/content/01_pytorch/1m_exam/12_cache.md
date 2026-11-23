---
title: "Cache: prefill, write, windowed read"
difficulty: Medium
---
# Task
Into layer `L` of the preallocated caches `kc` and `vc`:

1. prefill: write `k_pre` and `v_pre` (positions `0..T-1`);
2. write `k_new` and `v_new` at position `pos`;
3. read the live window for window `W` at `pos`: `lo` (first live position),
   `keys` and `vals` (`(B, HKV, n, DH)`), and `kv_pos` (the absolute positions
   of the keys you read).

# Given
```python
n_layers, B, HKV, DH, max_ctx, W, T = 2, 1, 2, 2, 16, 4, 6
kc = torch.zeros(n_layers, B, HKV, max_ctx, DH)
vc = torch.zeros(n_layers, B, HKV, max_ctx, DH)
L, pos = 1, T
k_pre, v_pre = torch.randn(B, HKV, T, DH), torch.randn(B, HKV, T, DH)
k_new, v_new = torch.randn(B, HKV, 1, DH), torch.randn(B, HKV, 1, DH)
```

# Starter
```python
TODO            # prefill kc
TODO            # prefill vc
TODO            # write k_new at pos
TODO            # write v_new at pos
lo = TODO
keys = TODO
vals = TODO
kv_pos = TODO
```

# Solution
```python
kc[L, :, :, :T, :] = k_pre
vc[L, :, :, :T, :] = v_pre
kc[L, :, :, pos, :] = k_new[:, :, 0, :]
vc[L, :, :, pos, :] = v_new[:, :, 0, :]
lo = max(0, pos - W + 1)
keys, vals = kc[L, :, :, lo:pos + 1, :], vc[L, :, :, lo:pos + 1, :]
kv_pos = torch.arange(lo, pos + 1)
```

# Checks
```python
check("lo", torch.tensor(lo), want=3)
check("keys", keys, shape=(B, HKV, W, DH))
check("kv_pos", kv_pos, want=torch.tensor([3, 4, 5, 6]))
check("the new key is last", keys[:, :, -1], want=k_new[:, :, 0])
check("prefilled key at 3 is there", keys[:, :, 0], want=k_pre[:, :, 3])
check("the other layer is untouched", kc[0].abs().sum(), want=0.0)
```

# Hints
1. 1k Drill 2: prefill is one slice assignment covering all `T` positions, and the read is one slice over the live range `[lo, pos]`, both ends included.
2. The window holds `W` positions ending at `pos`, clamped at 0. `k_new` is `(B, HKV, 1, DH)`: drop its length-1 axis to write at an integer `pos`, or write into a `pos:pos + 1` slice.

# Takeaway
`kv_pos` is what makes the mask work: the mask compares absolute positions,
and after a windowed read the keys no longer start at 0. Getting `lo` off by
one silently changes how much context the model sees.

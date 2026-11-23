---
title: Prefill, then the windowed read
difficulty: Medium
---
# Lesson
## The window

At position `pos` with window `W`, the live range is `[lo, pos]` where
`lo = max(0, pos - W + 1)`.

Getting `lo` off by one silently changes how much context the model sees.

# Task
1. Write a whole prompt's keys into `kc` in one assignment. `k_pre` is
   `(B, HKV, T, DH)` for positions `0..T-1`; they go into layer `0`.
2. Write `window_read(kc, pos, W)` returning `lo, keys, kv_pos`: `lo` as above
   (a Python int), `keys` the layer-0 keys in `[lo, pos]`, shape
   `(B, HKV, n, DH)` with `n = min(pos + 1, W)`, and `kv_pos`, the
   absolute positions of those keys, which the mask needs.

The checks call `window_read` at `pos = 2` and `pos = 7`.

# Given
```python
HKV, DH, B = 2, 2, 1
W, max_ctx, T = 4, 16, 6
kc = torch.zeros(1, B, HKV, max_ctx, DH)
k_pre = torch.randn(B, HKV, T, DH)
```

# Starter
```python
TODO                           # one assignment, all T positions


def window_read(kc, pos, W):
    lo = TODO
    keys = TODO
    kv_pos = TODO
    return lo, keys, kv_pos
```

# Solution
```python
kc[0, :, :, :T, :] = k_pre


def window_read(kc, pos, W):
    lo = max(0, pos - W + 1)
    keys = kc[0, :, :, lo:pos + 1, :]
    kv_pos = torch.arange(lo, pos + 1)
    return lo, keys, kv_pos
```

# Checks
```python
if not kc[0, :, :, :T, :].any():
    expect("prefill landed", False, "kc[0] (layer 0) is still all zeros at positions 0..T-1: nothing was written there. Is line 1 still TODO?")
else:
    check("prefill landed", kc[0, :, :, :T, :], want=k_pre)
check("nothing spilled past T", kc[0, :, :, T:, :].abs().sum(), want=0.0)

lo2, keys2, kv_pos2 = window_read(kc, 2, W)
check("pos=2 lo", lo2, want=0)
check("pos=2 keys", keys2, shape=(B, HKV, 3, DH))
check("pos=2 kv_pos", kv_pos2, want=torch.arange(0, 3))

lo7, keys7, kv_pos7 = window_read(kc, 7, W)
check("pos=7 lo", lo7, want=4)
check("pos=7 keys", keys7, shape=(B, HKV, 4, DH))
check("pos=7 kv_pos", kv_pos7, want=torch.arange(4, 8))
```

# Hints
1. A slice on the left of `=` writes a whole block at once: `kc[0, :, :, a:b, :] = ...` takes a tensor whose position axis has length `b - a`.
2. Check `lo` by hand: at `pos=7`, `W=4` the window is the four positions ending at 7. The slice end is exclusive, and `kv_pos` must start at `lo`, not at 0.

# Takeaway
`kv_pos` is what makes the mask from 1f work: the mask compares absolute
positions, and after a windowed read the keys no longer start at 0.

---
title: "B. Write the cache, read the window"
difficulty: Easy
---
# Lesson
## B. Write the cache, read the window

```text
cache_step(k, v, kc, vc, L, pos, cfg) -> keys, vals, kv_pos
```

Write into slot `pos`, then read `[lo, pos]` with
`lo = max(0, pos - window + 1)`, and return the keys' absolute positions.

# Task
Write `cache_step(k, v, kc, vc, L, pos, cfg)`. `k` and `v` are
`(1, HKV, 1, DH)`; `kc` and `vc` are `(n_layers, 1, HKV, max_ctx, DH)` from
`new_cache`. Write `k` and `v` into slot `pos` of layer `L` in place, then
return:

- `keys`, `vals`: layer `L`'s window `[lo, pos]`, each `(1, HKV, n, DH)`
- `kv_pos`: the absolute positions `lo..pos`, shape `(n,)`

with `lo = max(0, pos - cfg["window"] + 1)`.

# Given
```python
kc, vc = new_cache(CFG, 12)
```

# Starter
```python
def cache_step(k, v, kc, vc, L, pos, cfg):
    TODO                    # write k into kc at slot pos
    TODO                    # write v into vc at slot pos
    lo = TODO
    keys = TODO
    vals = TODO
    kv_pos = TODO
    return keys, vals, kv_pos
```

# Solution
```python
def cache_step(k, v, kc, vc, L, pos, cfg):
    kc[L, :, :, pos, :] = k[:, :, 0, :]
    vc[L, :, :, pos, :] = v[:, :, 0, :]
    lo = max(0, pos - cfg["window"] + 1)
    return kc[L, :, :, lo:pos+1, :], vc[L, :, :, lo:pos+1, :], torch.arange(lo, pos+1)
```

# Checks
```python
for p in range(7):
    kk = torch.full((1, CFG["HKV"], 1, CFG["DH"]), float(p + 1))
    vv = torch.full((1, CFG["HKV"], 1, CFG["DH"]), float(-(p + 1)))
    keys, vals, kv_pos = cache_step(kk, vv, kc, vc, 0, p, CFG)

check("keys", keys, shape=(1, CFG["HKV"], CFG["window"], CFG["DH"]))
check("kv_pos", kv_pos, want=torch.tensor([3, 4, 5, 6]))
check("window contents", keys[0, 0, :, 0], want=torch.tensor([4.0, 5.0, 6.0, 7.0]))
check("vals came from vc", vals[0, 0, :, 0], want=torch.tensor([-4.0, -5.0, -6.0, -7.0]))
check("layer 1 untouched", kc[1].abs().sum(), want=0.0)
_, _, p0 = cache_step(kk, vv, kc, vc, 1, 0, CFG)
check("early positions give a short window", p0, want=torch.tensor([0]))
```

# Hints
1. This is 1k again with a layer index: index the slot with plain integers `L` and `pos`, and drop `k`'s length-1 position axis on the way in.
2. Slice the window as `lo:pos+1` (the end is exclusive); `torch.arange` over the same range gives `kv_pos`.

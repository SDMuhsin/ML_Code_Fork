---
title: Write one slot, read the live window
difficulty: Easy
---
# Lesson
## Joining and splitting

| | |
|---|---|
| `torch.cat([a, b], dim=d)` | join along an existing axis; the `d` lengths add |
| `torch.stack([a, b], dim=d)` | join along a new axis of length 2 |
| `torch.split(t, [a, b, c], dim)` | uneven pieces; use this for qkv |

```python demo
a, b = torch.zeros(2, 3), torch.ones(2, 3)

print("cat(dim=0)  ", S(torch.cat([a, b], 0)))
print("cat(dim=1)  ", S(torch.cat([a, b], 1)))
print("stack(dim=0)", S(torch.stack([a, b], 0)), " a new axis")
```

## Two ways to keep a cache

A simple reference grows it every step:

```python
k_cache = torch.cat([k_cache, k_new], dim=2)     # allocates every step
```

A kernel cannot do that; it has no allocator. It preallocates once and writes
in place:

```python
k_cache[layer, :, :, pos, :] = k_new[:, :, 0, :]
keys = k_cache[layer, :, :, lo:pos+1, :]
```

These problems use the second form. It matches the runner, so a cache-indexing
bug in your reference lands in the same place as the kernel's.

```python demo
HKV, DH = 2, 2
n_layers, B, max_ctx = 2, 1, 6
kc = torch.zeros(n_layers, B, HKV, max_ctx, DH)

print("cache", S(kc), " (layer, batch, kv_head, position, head_dim)")

for pos in range(3):
    kc[0, :, :, pos, :] = float(pos + 1)
    print(f"  pos {pos}: live window {S(kc[0, :, :, :pos+1, :])}")

print("layer 0, kv head 0, dim 0:", kc[0, 0, 0, :, 0].tolist())
```

Writing in place changes every view of that storage. The cache relies on that,
and it is also how a tensor gets corrupted by accident.

# Task
Write `write_and_read(kc, vc, L, pos, k_new, v_new)`. At layer `L`, write
`k_new` and `v_new` (each `(B, HKV, 1, DH)`) into slot `pos` of the caches
`kc` and `vc` (each `(n_layers, B, HKV, max_ctx, DH)`), in place, then
return the live windows covering `0..pos`:

- `keys`: shape `(B, HKV, pos+1, DH)`
- `vals`: shape `(B, HKV, pos+1, DH)`

The checks call it at two different positions, `0` and `4`.

# Given
```python
HKV, DH = 2, 2
n_layers, B, max_ctx, L = 2, 1, 10, 1
kc = torch.zeros(n_layers, B, HKV, max_ctx, DH)
vc = torch.zeros(n_layers, B, HKV, max_ctx, DH)
```

# Starter
```python
def write_and_read(kc, vc, L, pos, k_new, v_new):
    TODO                       # write k_new into kc at slot pos
    TODO                       # write v_new into vc at slot pos
    keys = TODO                # (B, HKV, pos+1, DH)
    vals = TODO
    return keys, vals
```

# Solution
```python
def write_and_read(kc, vc, L, pos, k_new, v_new):
    kc[L, :, :, pos, :] = k_new[:, :, 0, :]
    vc[L, :, :, pos, :] = v_new[:, :, 0, :]
    keys = kc[L, :, :, :pos + 1, :]
    vals = vc[L, :, :, :pos + 1, :]
    return keys, vals
```

# Checks
```python
k_new0, v_new0 = torch.randn(B, HKV, 1, DH), torch.randn(B, HKV, 1, DH)
keys0, vals0 = write_and_read(kc, vc, L, 0, k_new0, v_new0)
check("pos=0 keys", keys0, shape=(B, HKV, 1, DH))
check("pos=0 vals", vals0, shape=(B, HKV, 1, DH))
check("pos=0 the new key landed at pos", keys0[:, :, 0, :], want=k_new0[:, :, 0, :])

k_new4, v_new4 = torch.randn(B, HKV, 1, DH), torch.randn(B, HKV, 1, DH)
keys4, vals4 = write_and_read(kc, vc, L, 4, k_new4, v_new4)
check("pos=4 keys", keys4, shape=(B, HKV, 5, DH))
check("pos=4 vals", vals4, shape=(B, HKV, 5, DH))
check("pos=4 the new key landed at pos", keys4[:, :, 4, :], want=k_new4[:, :, 0, :])

check("the other layer is untouched", kc[0].abs().sum(), want=0.0)
```

# Hints
1. Indexing on the left of `=` writes into the existing storage. Pick layer `L` and slot `pos` with plain integers and keep `:` for batch, head and head_dim.
2. `k_new` has a length-1 position axis that the cache slot does not have, so index it away with `[:, :, 0, :]`. The window is a slice that ends at `pos + 1`, because slice ends are exclusive.

# Takeaway
`k_new[:, :, 0, :]` and `k_new.squeeze(2)` are the same thing here. Prefer
the explicit index: `squeeze(2)` would silently do nothing if that axis were
ever not 1.

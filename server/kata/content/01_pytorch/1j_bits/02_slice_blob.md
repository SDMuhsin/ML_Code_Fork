---
title: Slice the blob apart
difficulty: Medium
---
# Lesson
## The layout

One row of a `blk1632_b4` matrix is a fixed byte range in three planes:

```text
row r starts at r * stride,  stride = 3N/8 bytes

  [0,    N/8)   mask    one bit per column, lowest first
  [N/8, 3N/8)   codes   N/4 bytes: the surviving 4-bit codes, two per byte
```

Exactly 16 of every aligned 32 columns survive, a fixed count per block. So
block `t`'s codes always start at byte `8t`, and any thread can jump straight
there with no table and no counting.

The scales live in separate arrays:

- `scale`: fp16, one per group of 128 input columns per row
- `zero`: uint8, same shape
- `col_scale`: fp16, one per input column

Each is a separate array so it can be read as its own contiguous stream.

`make_blob(K, N, group, seed)` (defined for this whole module) builds a
miniature packed matrix, as raw bytes, the way a file looks. It returns
`(blob, keep, codes, scale, zero)`.

```python demo
K, N, G = 4, 32, 8
BLOB, KEEP, CODE, SCALE, ZERO = make_blob(K, N, G)

print("blob", S(BLOB), BLOB.dtype, " one opaque byte array")
print("survivors in the first 32-column block of row 0:", int(KEEP[0, :32].sum()))
```

# Task
Cut `BLOB` into its four arrays. Compute the offsets from `K`, `N` and `G`.

`BLOB` is four planes back to back, each covering all `K` rows: the mask
bytes (`N/8` per row), the code bytes (`N/4` per row), the fp16 scales
(`N/G` per row, two bytes each), then the uint8 zeros (`N/G` per row).

- `o1`, `o2`, `o3`: the byte offsets where the mask, code and scale planes end
- `mask`: uint8, shape `(K, N/8)`
- `nib`: uint8, shape `(K, N/4)`
- `scale`: float16, shape `(K, N/G)`
- `zero`: uint8, shape `(K, N/G)`

The scales are fp16 values stored as bytes. Reinterpret them with
`view(dtype)` from 1c; do not convert them.

# Given
```python
K, N, G = 4, 32, 8
BLOB = make_blob(K, N, G)[0]
```

# Starter
```python
o1 = TODO           # end of the mask plane
o2 = TODO           # end of the code plane
o3 = TODO           # end of the fp16 scales
mask = TODO
nib = TODO
scale = TODO        # reinterpret the bytes
zero = TODO
```

# Solution
```python
o1 = K * N // 8
o2 = o1 + K * N // 4
o3 = o2 + K * (N // G) * 2
mask = BLOB[:o1].reshape(K, N // 8)
nib = BLOB[o1:o2].reshape(K, N // 4)
scale = BLOB[o2:o3].view(torch.float16).reshape(K, N // G)
zero = BLOB[o3:].reshape(K, N // G)
```

# Checks
```python
_, KEEP, CODE, SCALE, ZERO = make_blob(K, N, G)   # the arrays BLOB was packed from

check("mask", mask, shape=(K, N // 8), dtype=torch.uint8)
check("nib", nib, shape=(K, N // 4), dtype=torch.uint8)
check("scale", scale, shape=(K, N // G), dtype=torch.float16)
check("scale values", scale.float(), want=SCALE.float())
check("zero values", zero.float(), want=ZERO.float())
```

# Hints
1. Each offset is the previous one plus the size in bytes of one plane: rows times bytes per row. Then each array is a slice of `BLOB` reshaped to `(K, ...)`.
2. An fp16 takes two bytes, so the scale plane is `K * (N // G) * 2` bytes long. `.view(torch.float16)` on that byte slice reinterprets each pair as one half; `.to(torch.float16)` would convert each byte instead.

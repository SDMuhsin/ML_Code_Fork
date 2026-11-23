---
title: Tiled matvec
difficulty: Medium
---
# Lesson
## Tiles

A 4096x4096 matrix does not fit in a GPU's fast on-chip memory, so a matmul
loops over tiles and accumulates.

```python demo
A = torch.randn(8, 12)
B = torch.randn(12, 6)
out = torch.zeros(8, 6)

for m0 in range(0, 8, 4):
    for n0 in range(0, 6, 2):
        acc = torch.zeros(4, 2)
        for k0 in range(0, 12, 3):              # the reduction loop
            acc += A[m0:m0+4, k0:k0+3] @ B[k0:k0+3, n0:n0+2]
        out[m0:m0+4, n0:n0+2] = acc

print("matches A @ B:", torch.allclose(out, A @ B, atol=1e-5))
```

The `k0` loop is why a kernel needs an accumulator separate from its output.
Tile size changes the memory traffic and not the answer, except in floating
point, where it changes the summation order.

# Task
Write `tiled_matvec(W, x, R, C)` to compute `W @ x` touching `W` only through
`(R, C)` tiles: `R` output rows and `C` input columns at a time.

In the kernel, `R` is a tuning parameter, and it must not change the answer.
The last tile in each direction may be short.

The checks call it on `W` of shape `(7, 11)`, divisible by neither `R` nor `C`.

# Given
```python
W = torch.randn(7, 11)          # divisible by neither R nor C
x = torch.randn(11)
```

# Starter
```python
def tiled_matvec(W, x, R, C):
    K, N = W.shape
    out = torch.zeros(K)
    for r0 in range(0, K, R):
        r1 = TODO               # end of this row group, clamped to K
        acc = TODO              # accumulator for this row group
        for c0 in range(0, N, C):
            c1 = TODO
            acc = TODO          # add this tile's contribution
        out[r0:r1] = acc
    return out
```

# Solution
```python
def tiled_matvec(W, x, R, C):
    K, N = W.shape
    out = torch.zeros(K)
    for r0 in range(0, K, R):
        r1 = min(r0 + R, K)
        acc = torch.zeros(r1 - r0)
        for c0 in range(0, N, C):
            c1 = min(c0 + C, N)
            acc += W[r0:r1, c0:c1] @ x[c0:c1]
        out[r0:r1] = acc
    return out
```

# Checks
```python
check("R=1  C=1 ", tiled_matvec(W, x, 1, 1), want=W @ x)
check("R=2  C=3 ", tiled_matvec(W, x, 2, 3), want=W @ x)
check("R=4  C=5 ", tiled_matvec(W, x, 4, 5), want=W @ x)
check("R=7  C=11", tiled_matvec(W, x, 7, 11), want=W @ x)
check("R=16 C=32", tiled_matvec(W, x, 16, 32), want=W @ x)
```

# Hints
1. It is the demo's pattern without the `n0` loop, since `x` is a vector: a zeroed accumulator per row group, a loop over the columns adding each tile's product, then one write to `out`.
2. Clamp each tile end with `min(start + size, limit)`, so the last row group's accumulator is shorter too. A tile slices `W` on both axes, but `x` only on the column range.

# Takeaway
Both parts carry over to CUDA. `acc` is a fixed-size local array, which
becomes registers, and `out` is written once, after the column loop.

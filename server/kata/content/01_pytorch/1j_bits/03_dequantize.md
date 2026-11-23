---
title: Dequantize
difficulty: Medium
---
# Task
Write `dequant(mask, nib, scale, zero, col_scale, G)`, returning
`(keep, surv, q, out)`:

```text
W[k,j] = (code[k,j] - zero[k, j//G]) * scale[k, j//G] * col_scale[j] * keep[k,j]
```

- `keep`: `(K, N)` bool, unpacked from `mask` (lowest bit first)
- `surv`: `(K, N//2)`, the survivor codes in column order (low nibble first)
- `q`: `(K, N)` dense codes as float, 0 where pruned
- `out`: `(K, N)`, the dequantized `W`

Each step is checked separately so a mistake is easy to locate.

The order in the last step matters. Multiplying by `keep` after the affine
makes every pruned position exactly `0.0`, whatever `zero` holds.

# Given
```python
K, N, G = 4, 32, 8
torch.manual_seed(3)
BLOB2 = make_blob(K, N, G, seed=3)[0]
o1, o2 = K * N // 8, K * N // 8 + K * N // 4
mask2 = BLOB2[:o1].reshape(K, N // 8)
nib2 = BLOB2[o1:o2].reshape(K, N // 4)
scale2 = torch.rand(K, N // G) + 0.5
zero2 = torch.rand(K, N // G) * 16 + 0.37     # a value a uint8 could never hold
cs = torch.rand(N) + 0.5
```

# Starter
```python
def dequant(mask, nib, scale, zero, col_scale, G):
    K, nb = mask.shape
    N = nb * 8
    keep = TODO       # (K, N) bool
    surv = TODO       # (K, N//2) survivor codes in column order
    q = TODO          # (K, N) dense codes, 0 where pruned  (two lines is fine)
    out = TODO        # the group affine
    out = TODO        # times col_scale, times keep
    return keep, surv, q, out
```

# Solution
```python
def dequant(mask, nib, scale, zero, col_scale, G):
    K, nb = mask.shape
    N = nb * 8
    keep = ((mask[:, :, None] >> torch.arange(8, dtype=torch.uint8)) & 1).reshape(K, N).bool()
    surv = torch.stack([nib & 0xF, (nib >> 4) & 0xF], dim=-1).reshape(K, N // 2)
    q = torch.zeros(K, N)
    q[keep] = surv.reshape(-1).float()
    out = ((q.view(K, N // G, G) - zero[:, :, None]) * scale[:, :, None]).reshape(K, N)
    out = out * col_scale[None, :] * keep.float()
    return keep, surv, q, out
```

# Checks
```python
_, keep_ref, code_ref, _, _ = make_blob(K, N, G, seed=3)   # what BLOB2 was packed from
keep, surv, q, Wd = dequant(mask2, nib2, scale2, zero2, cs, G)

check("step 1 keep", keep.int(), want=keep_ref.int())
check("step 2 survivor codes", surv, shape=(K, N // 2))
check("step 3 dense codes", q, want=code_ref.float())
check("pruned are exactly 0.0", Wd[~keep_ref],
      want=torch.zeros(int((~keep_ref).sum())), atol=0.0, rtol=0.0)
check("values", Wd, want=((code_ref.float() - zero2.repeat_interleave(G, 1))
                          * scale2.repeat_interleave(G, 1)
                          * cs[None, :] * keep_ref.float()))
```

# Hints
1. `keep` is the bitmap unpack from this module's first lesson. For `surv`, each byte of `nib` holds two codes: the low nibble is the earlier column.
2. Stack the low and high nibbles on a new last axis before reshaping, so they interleave. Scatter the codes into a zeros tensor with `q[keep] = ...`; for the affine, view `q` as `(K, N//G, G)` so each group's `zero` and `scale` broadcast with a blank last axis.

# Takeaway
This is the core of the real weight reader. The real one adds a third bit-plane for 6-bit codes, all planes sliced
from one fixed-stride row, and fused matrices whose `col_scale` is `[2, N]`
or `[3, N]` because each constituent has its own.

## Fused matmuls and col_scale

Fusing the weights of q, k and v means stacking their rows. The activation
is harder. Each constituent has its own `col_scale`,
and it cannot be folded into the group scale because it varies within a
group. So the kernel folds it into the activation instead, and `qkv` needs
three column-scaled copies of `h`.

```python
D = 8
cs3 = torch.rand(3, D) + 0.5
h = torch.randn(1, D)

print("h", S(h), " col_scale", S(cs3), " -> staged", S(h * cs3))
print("q reads row 0, k row 1, v row 2.")
# h (1, 8)  col_scale (3, 8)  -> staged (3, 8)
```

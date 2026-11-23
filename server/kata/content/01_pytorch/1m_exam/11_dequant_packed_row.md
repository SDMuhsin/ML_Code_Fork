---
title: Dequantize a packed row
difficulty: Hard
---
# Task
Write `read_and_dequant(blob, K, N, G, col_scale)`. `blob` is the raw byte
array from `make_blob`: the mask plane (one bit per column, lowest first), the
code plane (surviving 4-bit codes, two per byte, low nibble first), the fp16
group scales stored as bytes, then the uint8 zeros. Slice it apart, then
dequantize:

```text
W[k,j] = (code[k,j] - zero[k,j//G]) * scale[k,j//G] * col_scale[j] * keep[k,j]
```

Apply `keep` last so pruned positions are exactly 0.0.

Return `keep` `(K, N)` bool, `q` `(K, N)` the dense codes (0 where pruned),
and `out` `(K, N)` the dequantized weights.

# Given
```python
def make_blob(K, N, group, seed):
    g = gen(seed)
    keep = torch.zeros(K, N, dtype=torch.bool)
    for r in range(K):
        for t in range(N // 32):
            keep[r, t * 32 + torch.randperm(32, generator=g)[:16]] = True
    code = torch.randint(0, 16, (K, N), generator=g)
    code[~keep] = 0
    mask = ((keep.int().reshape(K, N // 8, 8) << torch.arange(8)).sum(-1)).to(torch.uint8)
    nib = (code[keep].reshape(K, N // 2)[:, 0::2] |
           (code[keep].reshape(K, N // 2)[:, 1::2] << 4)).to(torch.uint8)
    sc = (torch.rand(K, N // group, generator=g) + 0.5).half()
    zr = torch.randint(0, 16, (K, N // group), generator=g).to(torch.uint8)
    blob = torch.cat([mask.reshape(-1), nib.reshape(-1),
                      sc.reshape(-1).view(torch.uint8), zr.reshape(-1)])
    return blob, keep, code, sc, zr


K, N, G = 6, 32, 8
blob, keep_ref, code_ref, sc_ref, zr_ref = make_blob(K, N, G, seed=7)
cs = torch.rand(N) + 0.5
```

# Starter
```python
def read_and_dequant(blob, K, N, G, col_scale):
    o1 = TODO
    o2 = TODO
    o3 = TODO
    mask = TODO
    nib = TODO
    sc = TODO           # fp16 bytes -> float
    zr = TODO
    keep = TODO
    surv = TODO
    q = TODO
    out = TODO          # group affine
    out = TODO          # times col_scale, times keep
    return keep, q, out
```

# Solution
```python
def read_and_dequant(blob, K, N, G, col_scale):
    o1 = K * N // 8
    o2 = o1 + K * N // 4
    o3 = o2 + K * (N // G) * 2
    mask = blob[:o1].reshape(K, N // 8)
    nib = blob[o1:o2].reshape(K, N // 4)
    sc = blob[o2:o3].view(torch.float16).reshape(K, N // G).float()
    zr = blob[o3:].reshape(K, N // G).float()
    keep = ((mask[:, :, None] >> torch.arange(8, dtype=torch.uint8)) & 1).reshape(K, N).bool()
    surv = torch.stack([nib & 0xF, (nib >> 4) & 0xF], dim=-1).reshape(K, N // 2)
    q = torch.zeros(K, N)
    q[keep] = surv.reshape(-1).float()
    out = ((q.view(K, N // G, G) - zr[:, :, None]) * sc[:, :, None]).reshape(K, N)
    out = out * col_scale[None, :] * keep.float()
    return keep, q, out
```

# Checks
```python
keep, q, Wd = read_and_dequant(blob, K, N, G, cs)
check("keep", keep.int(), want=keep_ref.int())
check("16 survivors per 32-column block", keep[:, :32].sum(-1).double(),
      want=torch.full((K,), 16.0), atol=0.0, rtol=0.0)
check("dense codes", q, want=code_ref.float())
check("pruned are exactly 0.0", Wd[~keep_ref],
      want=torch.zeros(int((~keep_ref).sum())), atol=0.0, rtol=0.0)
check("values", Wd, want=((code_ref.float() - zr_ref.float().repeat_interleave(G, 1))
                          * sc_ref.float().repeat_interleave(G, 1)
                          * cs[None, :] * keep_ref.float()), atol=1e-3)
```

# Hints
1. 1j Drill 2 slices a blob like this one apart (the scales are fp16 values stored as bytes: reinterpret with `view`, do not convert); 1j Drill 3 is the dequant itself.
2. The planes are `K*N/8` bytes of mask, `K*N/4` of codes, `K*(N/G)*2` of fp16 scales, then the zeros. Unpack bits as in 1j "Unpacking a bitmap without a loop"; each code byte holds two codes, low nibble first; `q[keep] = ...` scatters survivors in column order.

# Takeaway
Pruned weights must dequantize to exactly 0.0. Multiplying by `keep` after
the affine guarantees that, whatever `zero` holds.

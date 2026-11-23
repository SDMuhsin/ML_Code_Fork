---
title: Fuse, project, split
difficulty: Medium
---
# Task
The packed format stores q, k and v as one matrix, rows stacked:
`[Wq; Wk; Wv]`.

Build it, do a single projection of one decode token, and split the result
into per-head `q`, `k`, `v`:

- `Wqkv`: the fused weight, shape `(H*DH + 2*HKV*DH, D)`
- `qkv`: one projection of `h` through `Wqkv`, shape `(1, 1, H*DH + 2*HKV*DH)`
- `q`: shape `(1, H, 1, DH)`
- `k`, `v`: shape `(1, HKV, 1, DH)` each

# Given
```python
H, DH, D = 4, 2, 8
HKV = 2
h = torch.randn(1, 1, D)
Wq = torch.randn(H * DH, D)
Wk = torch.randn(HKV * DH, D)
Wv = torch.randn(HKV * DH, D)
```

# Starter
```python
Wqkv = TODO

qkv = TODO          # one projection of h
q = TODO            # (1, H, 1, DH)
k = TODO            # (1, HKV, 1, DH)
v = TODO            # (1, HKV, 1, DH)
```

# Solution
```python
Wqkv = torch.cat([Wq, Wk, Wv], dim=0)

qkv = F.linear(h, Wqkv)
nq, nk = H * DH, HKV * DH
q = qkv[..., :nq].reshape(1, 1, H, DH).transpose(1, 2)
k = qkv[..., nq:nq + nk].reshape(1, 1, HKV, DH).transpose(1, 2)
v = qkv[..., nq + nk:].reshape(1, 1, HKV, DH).transpose(1, 2)
```

# Checks
```python
check("Wqkv", Wqkv, shape=(H * DH + 2 * HKV * DH, D))
check("Wqkv starts with the rows of Wq", Wqkv[:H * DH], want=Wq)
check("qkv", qkv, shape=(1, 1, H * DH + 2 * HKV * DH))
check("q", q, shape=(1, H, 1, DH),
      want=F.linear(h, Wq).view(1, 1, H, DH).transpose(1, 2), atol=1e-4)
check("k", k, shape=(1, HKV, 1, DH),
      want=F.linear(h, Wk).view(1, 1, HKV, DH).transpose(1, 2), atol=1e-4)
check("v", v, shape=(1, HKV, 1, DH),
      want=F.linear(h, Wv).view(1, 1, HKV, DH).transpose(1, 2), atol=1e-4)
```

# Hints
1. Stacking rows is a concatenation along axis 0. The weights are in `(out, in)` layout, so the projection is `F.linear`, not a bare `@`.
2. The last axis of `qkv` holds q's `H*DH` values, then k's `HKV*DH`, then v's. Slice each piece, reshape to `(1, 1, heads, DH)`, then swap the token and head axes.

# Takeaway
One matmul instead of three means one weight stream, and `h` is read once.

Each of q, k and v has its own per-input-column scale, though, so a fused
matmul needs three column-scaled copies of `h`. The real kernel has a whole
phase that writes them. Module 1j comes back to this.

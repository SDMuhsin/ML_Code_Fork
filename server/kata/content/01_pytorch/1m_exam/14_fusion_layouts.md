---
title: Two fusion layouts
difficulty: Medium
---
# Task
`qkv` is row-concatenated. `gateup` is row-interleaved: row `2i` is
gate row `i`, row `2i+1` is up row `i`. Build both and write the splitters.

- `Wqkv`: `Wq`, `Wk`, `Wv` fused row-concatenated, `(H*DH + 2*HKV*DH, D)`.
- `Wgu`: `Wg`, `Wu` fused row-interleaved, `(2I, D)`.
- `split_qkv(y, nq, nk)`: from `y = F.linear(h, Wqkv)`, return the `(q, k, v)` projections (`nq` is the q width, `nk` the k and v width).
- `split_gateup(y)`: from `y = F.linear(h, Wgu)`, return the `(gate, up)` projections.

# Given
```python
H, HKV, DH, D, I = 4, 2, 2, 8, 16
Wq, Wk, Wv = torch.randn(H * DH, D), torch.randn(HKV * DH, D), torch.randn(HKV * DH, D)
Wg, Wu = torch.randn(I, D), torch.randn(I, D)
```

# Starter
```python
Wqkv = TODO
Wgu = TODO


def split_qkv(y, nq, nk):
    return TODO, TODO, TODO


def split_gateup(y):
    return TODO, TODO
```

# Solution
```python
Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
Wgu = torch.stack([Wg, Wu], dim=1).reshape(2 * I, D)


def split_qkv(y, nq, nk):
    return y[..., :nq], y[..., nq:nq + nk], y[..., nq + nk:]


def split_gateup(y):
    return y[..., 0::2], y[..., 1::2]
```

# Checks
```python
check("Wqkv", Wqkv, shape=(H * DH + 2 * HKV * DH, D))
check("Wgu", Wgu, shape=(2 * I, D))
check("Wqkv row 0 is q row 0", Wqkv[0], want=Wq[0])
check("Wqkv first k row", Wqkv[H * DH], want=Wk[0])
check("Wgu row 0 is gate row 0", Wgu[0], want=Wg[0])
check("Wgu row 1 is up row 0", Wgu[1], want=Wu[0])
check("Wgu row 2 is gate row 1", Wgu[2], want=Wg[1])
h = torch.randn(2, 1, D)
a, b, c = split_qkv(F.linear(h, Wqkv), H * DH, HKV * DH)
check("split_qkv q", a, want=F.linear(h, Wq), atol=1e-4)
check("split_qkv k", b, want=F.linear(h, Wk), atol=1e-4)
check("split_qkv v", c, want=F.linear(h, Wv), atol=1e-4)
g_, u_ = split_gateup(F.linear(h, Wgu))
check("split_gateup gate", g_, want=F.linear(h, Wg), atol=1e-4)
check("split_gateup up", u_, want=F.linear(h, Wu), atol=1e-4)
```

# Hints
1. 1h Drill 2 fuses `[Wq; Wk; Wv]` with `cat` and splits the projection by slicing its last axis; 1b Drill 2 splits an interleaved `gu` with step slices.
2. For `Wgu`, put gate and up side by side on a new axis right after the row axis, then `reshape` so the rows alternate. Output feature `r` of `F.linear` comes from weight row `r`, so each splitter slices the last axis the same way the rows were laid out.

# Takeaway
`qkv` is row-concatenated; `gateup` is row-interleaved. Slicing an
interleaved output as if it were concatenated gives the right shapes and the
wrong numbers.

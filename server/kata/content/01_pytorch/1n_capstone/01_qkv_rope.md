---
title: "A. Project and rotate"
difficulty: Medium
---
# Lesson
## What you are building

```text
token -> embed -> [ norm -> attention -> +residual
                    norm -> MLP       -> +residual ] x N -> norm -> lm_head
```

The dimensions are toy-sized; the structure is BioMistral's. Setting the
eight size constants in `CFG` to the real values gives the real model.

```python demo
print(CFG)
```

## A naming trap

In a Mistral checkpoint the weight called `post_attention_layernorm` is the
norm applied before the MLP. The layer is

```text
h2 = h  + Attention( norm(h,  input_layernorm) )
h3 = h2 + MLP(       norm(h2, post_attention_layernorm) )
```

There are two norms per layer, each applied to its block's input. The code
below calls the second one `ln2` to avoid the misleading name.

The module setup builds the weights `W = make_weights(CFG)` and gives you
`new_cache`:

```python
def make_weights(cfg, seed=0):
    g = gen(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3
    D, H, HKV, DH, I, V = (cfg["D"], cfg["H"], cfg["HKV"], cfg["DH"], cfg["I"], cfg["V"])
    return dict(
        embed=r(V, D), lm_head=r(V, D),
        final_norm=torch.rand(D, generator=g) + 0.5,
        layers=[dict(
            ln1=torch.rand(D, generator=g) + 0.5,
            wq=r(H * DH, D), wk=r(HKV * DH, D), wv=r(HKV * DH, D), wo=r(D, H * DH),
            ln2=torch.rand(D, generator=g) + 0.5,
            wg=r(I, D), wu=r(I, D), wd=r(D, I),
        ) for _ in range(cfg["n_layers"])])


def new_cache(cfg, max_ctx, dtype=torch.float32):
    sh = (cfg["n_layers"], 1, cfg["HKV"], max_ctx, cfg["DH"])
    return torch.zeros(sh, dtype=dtype), torch.zeros(sh, dtype=dtype)
```

## The oracle

`naive_step` does the same computation with explicit Python loops in fp64. It
is slow and easy to verify by reading. Use it as a specification when you are
stuck. (It is in the module setup; part G checks you against it.)

```python
def naive_step(tok, pos, W, kc, vc, cfg):
    D, H, HKV, DH, I = cfg["D"], cfg["H"], cfg["HKV"], cfg["DH"], cfg["I"]
    G, win, eps, th, sc = cfg["G"], cfg["window"], cfg["eps"], cfg["theta"], cfg["scale"]
    # Weights as plain Python lists, read once: element-by-element tensor reads are slow on CUDA.
    embed, lm_head, final_norm = W["embed"].tolist(), W["lm_head"].tolist(), W["final_norm"].tolist()
    layers = [{n: t.tolist() for n, t in lw.items()} for lw in W["layers"]]
    h = [float(z) for z in embed[tok]]
    for L, lw in enumerate(layers):
        x = _rms(h, lw["ln1"], eps)
        qf, kf, vf = _mv(lw["wq"], x), _mv(lw["wk"], x), _mv(lw["wv"], x)
        for kvh in range(HKV):
            kc[L][kvh][pos] = _rope(kf[kvh*DH:(kvh+1)*DH], pos, DH, th)
            vc[L][kvh][pos] = vf[kvh*DH:(kvh+1)*DH]
        lo = max(0, pos - win + 1)
        attn = [0.0] * (H * DH)
        for hh in range(H):
            kvh = hh // G
            qh = _rope(qf[hh*DH:(hh+1)*DH], pos, DH, th)
            s = [sum(qh[d] * kc[L][kvh][j][d] for d in range(DH)) * sc
                 for j in range(lo, pos + 1)]
            mx = max(s)
            e = [math.exp(z - mx) for z in s]
            tot = sum(e)
            for d in range(DH):
                attn[hh*DH + d] = sum(e[n] * vc[L][kvh][lo+n][d] for n in range(len(e))) / tot
        o = _mv(lw["wo"], attn)
        h2 = [h[i] + o[i] for i in range(D)]
        y = _rms(h2, lw["ln2"], eps)
        gate, upv = _mv(lw["wg"], y), _mv(lw["wu"], y)
        act = [(gate[i] / (1.0 + math.exp(-gate[i]))) * upv[i] for i in range(I)]
        dn = _mv(lw["wd"], act)
        h = [h2[i] + dn[i] for i in range(D)]
    return torch.tensor(_mv(lm_head, _rms(h, final_norm, eps)), dtype=torch.float64)


def _rms(x, w, eps):
    r = 1.0 / math.sqrt(sum(float(z) ** 2 for z in x) / len(x) + eps)
    return [float(x[i]) * r * float(w[i]) for i in range(len(x))]


def _mv(W_, x):
    return [sum(float(W_[o][i]) * x[i] for i in range(len(x))) for o in range(len(W_))]


def _rope(vec, p, dh, theta):
    """The same rotation as rotate_half, index by index, written differently
    so that agreement is an independent check."""
    half = dh // 2
    out = [0.0] * dh
    for i in range(dh):
        ang = p / (theta ** (2.0 * (i % half) / dh))
        partner = -vec[i + half] if i < half else vec[i - half]
        out[i] = vec[i] * math.cos(ang) + partner * math.sin(ang)
    return out
```

## Helpers

The parts also use `rope_tables`, `rotate_half` and `apply_rope` from 1m, and
`repeat_kv` and `rms_norm`. All are in the module setup.

```python
def rope_tables(positions, head_dim, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def repeat_kv(x, n_rep):
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(b, hkv, n_rep, t, dh).reshape(b, hkv * n_rep, t, dh)


def rms_norm(x, w, eps=1e-5):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)
```

## A. Project and rotate

```text
qkv_rope(h, lw, cos, sin, cfg) -> q, k, v

h    : (1, 1, D), already normalised
cos  : (1, DH) tables for this one position
gives  q (1, H, 1, DH),  k (1, HKV, 1, DH),  v (1, HKV, 1, DH)
```

RoPE goes on q and k, not on v.

The tables arrive as an argument because the real reference builds them once
per step, not once per layer.

# Task
Write `qkv_rope(h, lw, cos, sin, cfg)` returning `q, k, v` as specified above:
project `h` through `lw["wq"]`, `lw["wk"]`, `lw["wv"]`, split the heads out,
cast the tables to `h.dtype`, and apply RoPE to `q` and `k` only.

# Given
```python
hx, lw0 = torch.randn(1, 1, CFG["D"]), W["layers"][0]
c0, s0 = rope_tables(torch.tensor([0]), CFG["DH"], CFG["theta"])
```

# Starter
```python
def qkv_rope(h, lw, cos, sin, cfg):
    H, HKV, DH = cfg["H"], cfg["HKV"], cfg["DH"]
    c, s = TODO, TODO       # cast the tables to h.dtype
    q = TODO                # (1, H, 1, DH)
    k = TODO                # (1, HKV, 1, DH)
    v = TODO                # (1, HKV, 1, DH)
    return TODO, TODO, TODO
```

# Solution
```python
def qkv_rope(h, lw, cos, sin, cfg):
    H, HKV, DH = cfg["H"], cfg["HKV"], cfg["DH"]
    c, s = cos.to(h.dtype), sin.to(h.dtype)
    q = F.linear(h, lw["wq"]).view(1, 1, H, DH).transpose(1, 2)
    k = F.linear(h, lw["wk"]).view(1, 1, HKV, DH).transpose(1, 2)
    v = F.linear(h, lw["wv"]).view(1, 1, HKV, DH).transpose(1, 2)
    return apply_rope(q, c, s), apply_rope(k, c, s), v
```

# Checks
```python
q, k, v = qkv_rope(hx, lw0, c0, s0, CFG)
check("q", q, shape=(1, CFG["H"], 1, CFG["DH"]))
check("k", k, shape=(1, CFG["HKV"], 1, CFG["DH"]))
check("v", v, shape=(1, CFG["HKV"], 1, CFG["DH"]))
check("at position 0 RoPE is the identity", q,
      want=F.linear(hx, lw0["wq"]).view(1, 1, CFG["H"], CFG["DH"]).transpose(1, 2))
check("v is not rotated", v,
      want=F.linear(hx, lw0["wv"]).view(1, 1, CFG["HKV"], CFG["DH"]).transpose(1, 2))

c5, s5 = rope_tables(torch.tensor([5]), CFG["DH"], CFG["theta"])
q5, k5, v5 = qkv_rope(hx, lw0, c5, s5, CFG)
expect("RoPE at position 5 differs from position 0", not torch.allclose(q5, q, atol=1e-5),
       "q at position 5 equals q at position 0: RoPE must differ from 0 at position 5")
check("v does not depend on position", v5, want=v)
check("q at position 5 is rotated", q5,
      want=apply_rope(F.linear(hx, lw0["wq"]).view(1, 1, CFG["H"], CFG["DH"]).transpose(1, 2), c5, s5))
check("k at position 5 is rotated too", k5,
      want=apply_rope(F.linear(hx, lw0["wk"]).view(1, 1, CFG["HKV"], CFG["DH"]).transpose(1, 2), c5, s5))
```

# Hints
1. `F.linear(h, w)` gives `(1, 1, heads * DH)`. Split the last axis into `(heads, DH)`, then swap the position and head axes so heads come second.
2. `.view(1, 1, H, DH).transpose(1, 2)` gives `(1, H, 1, DH)`. `apply_rope(x, c, s)` expects `x` in that `(batch, heads, positions, DH)` layout; `v` skips it.

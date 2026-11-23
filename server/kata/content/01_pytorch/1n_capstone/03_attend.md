---
title: "C. Attend"
difficulty: Medium
uses: [1n-1, 1n-2]
---
# Lesson
## C. Attend

```text
attend(q, keys, vals, lw, cfg) -> (1, 1, D)
```

Expand the kv heads, scale by `cfg["scale"]`, softmax over the key axis in
fp32, multiply by values, flatten the heads, project through `wo`.

No mask is needed here. At `Tq == 1` every key in `[lo, pos]` is allowed,
because the windowed read already excluded the future and everything too old.
This holds only because of the windowed read, and it is why a transposed mask
survives decode-only testing. Part H comes back to it.

# Task
Write `attend(q, keys, vals, lw, cfg)`. `q` is `(1, H, 1, DH)`; `keys` and
`vals` are `(1, HKV, n, DH)` from `cache_step`. Return `(1, 1, D)`: expand the
kv heads to `H` (use `repeat_kv`), take scaled scores, an fp32 softmax over the
key axis cast back to `q.dtype`, multiply by values, flatten the heads, then
project through `lw["wo"]`.

# Given
```python
hx, lw0 = torch.randn(1, 1, CFG["D"]), W["layers"][0]
c0, s0 = rope_tables(torch.tensor([0]), CFG["DH"], CFG["theta"])
kc, vc = new_cache(CFG, 12)
q, k, v = qkv_rope(hx, lw0, c0, s0, CFG)
keys, vals, _ = cache_step(k, v, kc, vc, 0, 0, CFG)
```

# Starter
```python
def attend(q, keys, vals, lw, cfg):
    H, DH, G = cfg["H"], cfg["DH"], cfg["G"]
    k = TODO                # expand kv heads to H
    v = TODO
    scores = TODO           # (1, H, 1, n), scaled
    probs = TODO            # fp32 softmax over the key axis, cast back
    out = TODO              # (1, 1, H*DH)
    return TODO             # through wo
```

# Solution
```python
def attend(q, keys, vals, lw, cfg):
    H, DH, G = cfg["H"], cfg["DH"], cfg["G"]
    k, v = repeat_kv(keys, G), repeat_kv(vals, G)
    scores = (q @ k.transpose(-1, -2)) * cfg["scale"]
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = (probs @ v).transpose(1, 2).reshape(1, 1, H * DH)
    return F.linear(out, lw["wo"])
```

# Checks
```python
oa = attend(q, keys, vals, lw0, CFG)
check("shape", oa, shape=(1, 1, CFG["D"]))
v0 = v.repeat_interleave(CFG["G"], dim=1).transpose(1, 2).reshape(1, 1, CFG["H"] * CFG["DH"])
check("with one key, attention returns that key's value", oa, want=F.linear(v0, lw0["wo"]))

q4 = torch.randn(1, CFG["H"], 1, CFG["DH"])
keys4, vals4 = torch.randn(1, CFG["HKV"], 4, CFG["DH"]), torch.randn(1, CFG["HKV"], 4, CFG["DH"])
check("with four keys, matches scaled grouped-query attention", attend(q4, keys4, vals4, lw0, CFG),
      want=F.linear(gqa_attention(q4, keys4, vals4, torch.ones(1, 4, dtype=torch.bool),
                                  CFG["scale"], CFG["G"]), lw0["wo"]))
```

# Hints
1. `q @ k.transpose(-1, -2)` gives `(1, H, 1, n)`: one score per head per key. The key axis is the last one, so that is where the softmax goes.
2. `F.softmax(..., dim=-1, dtype=torch.float32)` does the fp32 part. After `probs @ v` you have `(1, H, 1, DH)`: move the head axis next to `DH` before flattening to `H * DH`.

# Takeaway
The one-key check is a useful smoke test: softmax over a single key gives
weight 1.0, so the output is exactly V projected.

---
title: "F. One layer"
difficulty: Medium
uses: [1n-1, 1n-2, 1n-3, 1n-4]
---
# Lesson
## F. One layer

```text
h2 = h  + attention( norm(h,  ln1) )
h3 = h2 + mlp(       norm(h2, ln2) )
```

Attention here means `qkv_rope` then `cache_step` then `attend`.

# Task
Write `decoder_layer(h, lw, kc, vc, L, pos, cos, sin, cfg)` returning `h3`,
shape `(1, 1, D)`, as above. Use `rms_norm` with `cfg["eps"]`, and your
`qkv_rope`, `cache_step`, `attend` and `mlp` from the earlier parts.

# Given
```python
hx, lw0 = torch.randn(1, 1, CFG["D"]), W["layers"][0]
c0, s0 = rope_tables(torch.tensor([0]), CFG["DH"], CFG["theta"])
kc, vc = new_cache(CFG, 12)
```

# Starter
```python
def decoder_layer(h, lw, kc, vc, L, pos, cos, sin, cfg):
    q, k, v = TODO
    keys, vals, _ = TODO
    h2 = TODO
    return TODO
```

# Solution
```python
def decoder_layer(h, lw, kc, vc, L, pos, cos, sin, cfg):
    q, k, v = qkv_rope(rms_norm(h, lw["ln1"], cfg["eps"]), lw, cos, sin, cfg)
    keys, vals, _ = cache_step(k, v, kc, vc, L, pos, cfg)
    h2 = h + attend(q, keys, vals, lw, cfg)
    return h2 + mlp(rms_norm(h2, lw["ln2"], cfg["eps"]), lw)
```

# Checks
```python
oc = decoder_layer(hx, lw0, kc, vc, 0, 0, c0, s0, CFG)
check("shape", oc, shape=(1, 1, CFG["D"]))
expect("the layer changes h", not torch.allclose(oc, hx), "the output equals the input: the layer must change h")

# At position 0 there is one key, so attention is exactly v projected through wo.
x1 = rms_norm(hx, lw0["ln1"], CFG["eps"])
v1 = F.linear(x1, lw0["wv"]).view(1, CFG["HKV"], CFG["DH"]).repeat_interleave(CFG["G"], dim=1).reshape(1, 1, CFG["D"])
h2_want = hx + F.linear(v1, lw0["wo"])
y2 = rms_norm(h2_want, lw0["ln2"], CFG["eps"])
check("at position 0, matches the layer definition", oc,
      want=h2_want + F.linear(F.silu(F.linear(y2, lw0["wg"])) * F.linear(y2, lw0["wu"]), lw0["wd"]))

zero_lw = {k_: torch.zeros_like(v_) for k_, v_ in lw0.items()}
zero_lw["ln1"] = zero_lw["ln2"] = torch.ones(CFG["D"])
check("with wo and wd zeroed, the layer is the identity",
      decoder_layer(hx, zero_lw, kc, vc, 1, 0, c0, s0, CFG), want=hx)
```

# Hints
1. Each norm feeds only its own branch: the residual adds the branch output to the un-normalised input.
2. The first norm uses `lw["ln1"]` and is applied to `h`; the second uses `lw["ln2"]` and is applied to `h2`, the output of the attention residual.

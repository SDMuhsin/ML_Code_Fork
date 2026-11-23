---
title: "G. A full step"
difficulty: Medium
uses: [1n-1, 1n-2, 1n-3, 1n-4, 1n-5]
---
# Lesson
## G. A full step

```text
forward_step(tok, pos, W, kc, vc, cfg) -> (1, 1, V)
```

Embed, build the RoPE tables once, every layer in order, final norm,
`lm_head`. `tok` is a plain int, so choose the index rank that gives
`(1, 1, D)`.

# Task
Write `forward_step(tok, pos, W, kc, vc, cfg)` returning the logits,
`(1, 1, V)`: embed `tok` from `W["embed"]` as `(1, 1, D)`, build the RoPE
tables for position `pos` once, run every layer of `W["layers"]` in order with
your `decoder_layer`, then `rms_norm` with `W["final_norm"]` and project
through `W["lm_head"]`. The check compares you with the oracle `naive_step`.

# Given
```python
kc, vc = new_cache(CFG, 12)
nkc, nvc = naive_cache(CFG, 12)
ref = naive_step(3, 0, W, nkc, nvc, CFG)
```

# Starter
```python
def forward_step(tok, pos, W, kc, vc, cfg):
    h = TODO                # (1, 1, D)
    cos, sin = TODO         # built once, outside the layer loop
    for L, lw in enumerate(W["layers"]):
        h = TODO
    return TODO             # final norm, then lm_head
```

# Solution
```python
def forward_step(tok, pos, W, kc, vc, cfg):
    h = W["embed"][torch.tensor([[tok]])]
    cos, sin = rope_tables(torch.tensor([pos]), cfg["DH"], cfg["theta"])
    for L, lw in enumerate(W["layers"]):
        h = decoder_layer(h, lw, kc, vc, L, pos, cos, sin, cfg)
    return F.linear(rms_norm(h, W["final_norm"], cfg["eps"]), W["lm_head"])
```

# Checks
```python
lg = forward_step(3, 0, W, kc, vc, CFG)
check("shape", lg, shape=(1, 1, CFG["V"]))
print("yours :", [round(z, 5) for z in lg[0, 0].tolist()])
print("oracle:", [round(z, 5) for z in ref.tolist()])
check("matches the oracle", lg[0, 0].double(), want=ref)
```

# Hints
1. Indexing a `(V, D)` table with an integer tensor of shape `S` gives `S + (D,)`. Which index shape gives `(1, 1, D)`?
2. `rope_tables` takes a tensor of positions, here `[pos]`. The layer index `L` is what `decoder_layer` uses to pick its slice of the cache.

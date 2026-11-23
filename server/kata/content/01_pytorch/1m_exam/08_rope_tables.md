---
title: RoPE tables
difficulty: Medium
---
# Task
Write `rope_tables(positions, head_dim, theta=10000.0)` returning `(cos, sin)`,
each `(T, head_dim)` for `T = len(positions)`.

```text
inv_freq[i] = 1 / theta ** (2i / head_dim)      i = 0 .. head_dim/2 - 1
freqs[t, i] = positions[t] * inv_freq[i]
emb         = cat([freqs, freqs], dim=-1)       length head_dim
cos, sin    = emb.cos(), emb.sin()
```

`inv_freq` has `head_dim/2` entries; `cos` and `sin` have `head_dim`. The
duplication must be `cat`, not `repeat_interleave`.

# Starter
```python
def rope_tables(positions, head_dim, theta=10000.0):
    inv = TODO              # (head_dim/2,)
    freqs = TODO            # (T, head_dim/2)
    emb = TODO              # (T, head_dim)
    return TODO, TODO
```

# Solution
```python
def rope_tables(positions, head_dim, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()
```

# Checks
```python
pos = torch.arange(4)
cos, sin = rope_tables(pos, 8)
check("cos", cos, shape=(4, 8))
check("position 0 is all ones", cos[0], want=torch.ones(8))
check("position 0 sin is zero", sin[0], want=torch.zeros(8))
check("the halves are duplicated", cos[:, :4], want=cos[:, 4:])
check("column 0 is cos(pos)", cos[:, 0], want=torch.cos(pos.float()))
check("column 1 uses theta**(2/head_dim)", cos[:, 1],
      want=torch.cos(pos.float() / (10000.0 ** 0.25)))
```

# Hints
1. 1e "Outer products": a column times a row is an outer product, and that is how `freqs` is built.
2. `torch.arange(0, head_dim, 2)` gives every `2i` at once. Make `positions` float before the outer product.

# Takeaway
`cat` puts the same frequency at dims `i` and `i + head_dim/2`, the pair that
`rotate_half` rotates together in the next problem. `repeat_interleave` would
put it at `2i` and `2i+1` instead, a different pairing with the same shape.

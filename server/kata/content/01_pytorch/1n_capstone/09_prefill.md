---
title: "I. Prefill, and the bug it catches"
difficulty: Hard
uses: [1n-4]
---
# Lesson
## I. Prefill, and the bug it catches

Every part so far used `Tq == 1`. This part processes the whole prompt in one
pass.

```text
prefill(prompt, W, cfg, max_ctx) -> logits (1, T, V), kc, vc
```

Per layer:

1. norm, project to `q (1, H, T, DH)` and `k`/`v` `(1, HKV, T, DH)`
2. RoPE q and k with tables for all T positions
3. write all T positions into the cache in one assignment
4. build the `(T, T)` causal + window mask from `q_pos` and `kv_pos`
5. masked attention, then `wo`
6. residual, norm, MLP, residual

Step 4 is the reason for this part. With the two `None`s the wrong way round,
every check in A–H still passes.

The module setup gives you masked grouped-query attention:

```python
def gqa_attention(q, keys, vals, allowed, scale, n_rep):
    b, h, tq, dh = q.shape
    k, v = repeat_kv(keys, n_rep), repeat_kv(vals, n_rep)
    scores = ((q @ k.transpose(-1, -2)) * scale).masked_fill(
        ~allowed[None, None], float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return (probs @ v).transpose(1, 2).reshape(b, tq, h * dh)
```

# Task
Fill in the layer loop of `prefill` following steps 1–6. `allowed[i, j]` is
`True` iff query `i` may see key `j`: key position at or before the query's,
and within the last `cfg["window"]` positions. The checks compare every row
with the token-by-token oracle, and the cache with the oracle's.

# Given
```python
prompt = [4, 1, 7, 2, 9, 0]
nkc, nvc = naive_cache(CFG, 20)
oracle_rows = torch.stack([naive_step(t, p, W, nkc, nvc, CFG) for p, t in enumerate(prompt)])
oracle_k = torch.tensor(nkc, dtype=torch.float64)[:, None, :, :len(prompt)]   # (n_layers, 1, HKV, T, DH)
```

# Starter
```python
def prefill(prompt, W, cfg, max_ctx=20):
    T = len(prompt)
    kc, vc = new_cache(cfg, max_ctx, dtype=W["embed"].dtype)
    H, HKV, DH, G = cfg["H"], cfg["HKV"], cfg["DH"], cfg["G"]
    q_pos = torch.arange(T)
    cos, sin = rope_tables(q_pos, DH, cfg["theta"])
    h = W["embed"][torch.tensor(prompt)][None]
    for L, lw in enumerate(W["layers"]):
        x = rms_norm(h, lw["ln1"], cfg["eps"])
        q = TODO                # (1, H, T, DH)
        k = TODO                # (1, HKV, T, DH)
        v = TODO
        q = TODO                # roped
        k = TODO                # roped
        TODO                    # write all T key positions into kc
        TODO                    # and vc
        kv_pos = TODO
        allowed = TODO          # (T, T) causal AND window
        att = TODO              # via gqa_attention
        h2 = TODO
        h = TODO
    return F.linear(rms_norm(h, W["final_norm"], cfg["eps"]), W["lm_head"]), kc, vc
```

# Solution
```python
def prefill(prompt, W, cfg, max_ctx=20):
    T = len(prompt)
    kc, vc = new_cache(cfg, max_ctx, dtype=W["embed"].dtype)
    H, HKV, DH, G = cfg["H"], cfg["HKV"], cfg["DH"], cfg["G"]
    q_pos = torch.arange(T)
    cos, sin = rope_tables(q_pos, DH, cfg["theta"])
    h = W["embed"][torch.tensor(prompt)][None]
    for L, lw in enumerate(W["layers"]):
        x = rms_norm(h, lw["ln1"], cfg["eps"])
        q = F.linear(x, lw["wq"]).view(1, T, H, DH).transpose(1, 2)
        k = F.linear(x, lw["wk"]).view(1, T, HKV, DH).transpose(1, 2)
        v = F.linear(x, lw["wv"]).view(1, T, HKV, DH).transpose(1, 2)
        q = apply_rope(q, cos.to(x.dtype), sin.to(x.dtype))
        k = apply_rope(k, cos.to(x.dtype), sin.to(x.dtype))
        kc[L, :, :, :T, :] = k
        vc[L, :, :, :T, :] = v
        kv_pos = torch.arange(T)
        allowed = ((kv_pos[None, :] <= q_pos[:, None])
                   & (kv_pos[None, :] > q_pos[:, None] - cfg["window"]))
        att = gqa_attention(q, k, v, allowed, cfg["scale"], G)
        h2 = h + F.linear(att, lw["wo"])
        h = h2 + mlp(rms_norm(h2, lw["ln2"], cfg["eps"]), lw)
    return F.linear(rms_norm(h, W["final_norm"], cfg["eps"]), W["lm_head"]), kc, vc
```

# Checks
```python
pre_logits, pre_kc, pre_vc = prefill(prompt, W, CFG)
check("shape", pre_logits, shape=(1, len(prompt), CFG["V"]))
check("matches the token-by-token path", pre_logits[0].double(),
      want=oracle_rows, atol=1e-5)
check("the prefilled cache matches the decoded one",
      pre_kc[:, :, :, :len(prompt)], want=oracle_k, atol=1e-5)
```

# Hints
1. Steps 1–2 are part A with `T` positions instead of 1: `.view(1, T, heads, DH).transpose(1, 2)`, and `apply_rope` with the tables built above the loop.
2. In `allowed`, rows are queries and columns are keys: whichever position vector runs down the rows needs `[:, None]`, the other `[None, :]`. Sanity-check row 0: it may see key 0 and nothing else.

# Takeaway
Swap the two `None`s and every row is wrong, the last one included: each query
then sees itself and the next `window - 1` positions instead of the
previous ones. Drop the window half and the first `window` rows still match;
only the later ones move.

That is the bug A–H cannot see, and the reason the real verification runs a
1152-token prompt instead of a decode loop.

## Done

You have now written, from the definitions, embedding lookup, RMSNorm, fused
projections, NeoX RoPE, a preallocated sliding-window KV cache, grouped-query
attention, SwiGLU, greedy sampling, and the prefill path that has to agree
with all of it.

This reference is the specification the CUDA kernel must match. The later
lessons make the same arithmetic run faster.

Come back in a day or two and do the recall list on the lesson page.

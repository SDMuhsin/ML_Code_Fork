---
title: "H. Generate"
difficulty: Medium
uses: [1n-1, 1n-2, 1n-3, 1n-4, 1n-5, 1n-6]
---
# Lesson
## H. Generate, then prove the window slides

Feed the prompt one token at a time at positions 0, 1, 2, then take the
argmax and feed it back.

# Task
Complete `generate(prompt, n_new, W, cfg)`. Run `forward_step` on each prompt
token at positions `0, 1, 2, ...`, then `n_new` times pick the greedy next
token from the previous step's logits, append it to `ids`, and run one step on
it. Every step appends its logits row `lg[0, 0]` (shape `(V,)`) to `rows`.
It returns `ids` (prompt plus the new tokens, as Python ints) and the stacked
rows, `(len(prompt) + n_new, V)`.

# Given
```python
prompt = [4, 1, 7, 2, 9, 0]
```

# Starter
```python
def generate(prompt, n_new, W, cfg, max_ctx=20):
    kc, vc = new_cache(cfg, max_ctx, dtype=W["embed"].dtype)
    ids, rows = list(prompt), []
    pos = 0
    for tok in prompt:
        TODO            # one step; append lg[0, 0] to rows; advance pos
    for _ in range(n_new):
        nxt = TODO      # greedy pick from the previous logits
        ids.append(nxt)
        TODO            # one step on nxt; append; advance pos
    return ids, torch.stack(rows)
```

# Solution
```python
def generate(prompt, n_new, W, cfg, max_ctx=20):
    kc, vc = new_cache(cfg, max_ctx, dtype=W["embed"].dtype)
    ids, rows = list(prompt), []
    pos = 0
    for tok in prompt:
        rows.append(forward_step(tok, pos, W, kc, vc, cfg)[0, 0])
        pos += 1
    for _ in range(n_new):
        nxt = int(rows[-1].argmax(-1))
        ids.append(nxt)
        rows.append(forward_step(nxt, pos, W, kc, vc, cfg)[0, 0])
        pos += 1
    return ids, torch.stack(rows)
```

# Checks
```python
ids, rows = generate(prompt, 6, W, CFG)
nkc, nvc = naive_cache(CFG, 20)
oracle_rows = torch.stack([naive_step(t, p, W, nkc, nvc, CFG) for p, t in enumerate(ids)])

check("logits per step", rows, shape=(len(prompt) + 6, CFG["V"]))
check("the prompt is preserved", torch.tensor(ids[:len(prompt)]), want=torch.tensor(prompt))
check("matches the oracle every step", rows.double(), want=oracle_rows)
check("same greedy tokens", rows.argmax(-1), want=oracle_rows.argmax(-1))
print("ids:", ids)
```

# Hints
1. One cache serves the whole run, and `pos` goes up by one after every step, prompt or generated.
2. The next token is the argmax of the last row in `rows`; `int(...)` makes it a plain Python int that `forward_step` can embed.

# Takeaway
The generated tail is one token repeated. That is expected with random
weights, because greedy decoding from an untrained model falls into a fixed
point. This problem tests the plumbing.

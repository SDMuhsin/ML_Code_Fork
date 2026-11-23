---
title: Name the mask bug
difficulty: Medium
---
# Lesson
`attn_C` is missing half of its mask condition. Does it
disagree with the oracle at `Tq == 1`? And at `Tq == 3`?

# Task
Compare `attn_C` with `oracle` (both with `n_rep = G`, `window = 4`) and set:

- `differs_at_1`: a bool, `True` iff they disagree (beyond `atol=1e-5`) on the
  single query `q1` at position `q_pos1`
- `differs_at_3`: a bool, `True` iff they disagree on the full three-query
  `args`, whose oracle output is `ref`

# Given
```python
args = setup()
q, kk, vv, q_pos, kv_pos = args
ref = oracle(*args, G, 4)
q1, q_pos1 = q[:, :, -1:, :], q_pos[-1:]
```

# Starter
```python
differs_at_1 = TODO       # bool
differs_at_3 = TODO       # bool
```

# Solution
```python
differs_at_1 = not torch.allclose(attn_C(q1, kk, vv, q_pos1, kv_pos, G, 4),
                                  oracle(q1, kk, vv, q_pos1, kv_pos, G, 4), atol=1e-5)
differs_at_3 = not torch.allclose(attn_C(*args, G, 4), ref, atol=1e-5)
```

# Checks
```python
check("differs at Tq == 1", differs_at_1, want=False)
check("differs at Tq == 3", differs_at_3, want=True)
```

# Hints
1. `torch.allclose(a, b, atol=1e-5)` says whether two tensors agree; "differs" is its negation.
2. For `Tq == 1`, give both functions the single query `q1` and its position `q_pos1`; the keys, values and `kv_pos` stay the full ones. For `Tq == 3`, `ref` is already the oracle's answer.

# Takeaway
`attn_C` keeps `j > p - W` and forgets `j <= p`. At `Tq == 1` that is
indistinguishable from correct, because the windowed cache read already
contains nothing from the future. At `Tq == 3`, query 3 can see keys 4 and 5.

Several bugs in this project behave that way, correct at decode and broken in
prefill. That is why the real verification runs a 1152-token prompt instead of
only a decode loop.

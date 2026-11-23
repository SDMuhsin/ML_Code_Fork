---
title: Name the GQA bug
difficulty: Medium
---
# Lesson
## Name the two bugs

Each bug is found with one perturbation of the input. This problem finds the
GQA bug; the next one finds the mask bug.

For the GQA bug, pass keys and values in which only kv head 0 is nonzero.
Zero both, or the zeroed heads still contribute values. A correct implementation
then affects query heads `0..G-1` and no others.

# Task
Using the perturbed `pk`, `pv` from Given, run `attn_B` (with `n_rep = G` and
`window = 4`, as before) and find which query heads it lets kv head 0 affect:

- `out_B`: the output of `attn_B`, shape `(1, Tq, H*DH)`
- `per_head`: `(H,)`, the total magnitude per query head
- `heads_touched`: a sorted list of ints, the query heads whose magnitude is
  above `1e-6`

# Given
```python
q, kk, vv, q_pos, kv_pos = setup()
pk, pv = torch.zeros_like(kk), torch.zeros_like(vv)
pk[:, 0], pv[:, 0] = kk[:, 0], vv[:, 0]
```

# Starter
```python
out_B = TODO
per_head = TODO            # (H,) total magnitude per query head
heads_touched = TODO       # sorted list of ints
```

# Solution
```python
out_B = attn_B(q, pk, pv, q_pos, kv_pos, G, 4)
per_head = out_B.reshape(1, q.shape[2], H, DH).abs().sum((0, 1, 3))
heads_touched = sorted(torch.nonzero(per_head > 1e-6).flatten().tolist())
```

# Checks
```python
print("attn_B, kv head 0 only -> query heads affected:", heads_touched)
print("a correct implementation would give:", list(range(G)))
check("out_B", out_B, shape=(1, q.shape[2], H * DH))
check("per_head", per_head, shape=(H,))
check("heads_touched", torch.tensor(heads_touched),
      want=torch.tensor([j for j in range(H) if j % HKV == 0]))
```

# Hints
1. The last axis of `out_B` is the query heads laid side by side, `DH` numbers each. Reshape it so heads get their own axis, then sum `abs()` over every other axis.
2. `torch.nonzero(mask)` gives the indices where a boolean mask is true; flatten it and `.tolist()` it for a list of ints.

# Takeaway
B maps kv0 to q0 and q2, not q0 and q1. That is `repeat` where the mapping
needed `expand` then `reshape`.

---
title: The attention reshape, both ways
difficulty: Medium
---
# Task
1. From `q_flat` of shape `(B, T, H*DH)` make `q` of shape `(B, H, T, DH)`.
   Heads go in front of time, because attention batches over `(B, H)`.
2. From `o` of shape `(B, H, T, DH)` make `o_flat` of shape `(B, T, H*DH)`.
   Here `o` is your `q` (the line `o = q` is already in the starter), so
   a correct `o_flat` gives back `q_flat` exactly.

One of the two needs `reshape` rather than `view`. Find out which.

# Given
```python
B, T, H, DH = 2, 3, 4, 2
q_flat = torch.arange(B * T * H * DH).float().reshape(B, T, H * DH)
```

# Starter
```python
q = TODO

o = q
o_flat = TODO
```

# Solution
```python
q = q_flat.view(B, T, H, DH).transpose(1, 2)

o = q
o_flat = o.transpose(1, 2).reshape(B, T, H * DH)
```

# Checks
```python
check("q shape", q, shape=(B, H, T, DH))
check("q maps the right elements", q[1, 2, 0, :], want=q_flat[1, 0, 2 * DH:3 * DH])
check("o_flat shape", o_flat, shape=(B, T, H * DH))
check("round trip is exact", o_flat, want=q_flat, atol=0.0, rtol=0.0)
```

# Hints
1. Split the last axis into `(H, DH)` first, then swap the `T` and `H` axes. Coming out, undo those two steps in reverse order.
2. After `transpose`, the tensor is no longer contiguous, so merging `(H, DH)` back into one axis cannot be done with `view`.

# Takeaway
Going in, `view` works. Coming out, `reshape` is required, because the
transpose left the tensor non-contiguous.

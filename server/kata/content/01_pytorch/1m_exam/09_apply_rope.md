---
title: Apply RoPE
difficulty: Hard
uses: [1m-8]
---
# Task
Write `rotate_half(x)` and `apply_rope(x, cos, sin)`.

```text
rotate_half(x) = cat([-x[..., dh/2:], x[..., :dh/2]], dim=-1)
out            = x * cos + rotate_half(x) * sin
```

`x` is `(B, Hany, T, DH)`; `cos` and `sin` are `(T, DH)`. Place the blank
axes so it works for both `H` and `HKV` heads unchanged.

Your `rope_tables` from 1m-8 is available (the reference one if you have not
solved it yet).

# Given
```python
H, HKV, DH = 4, 2, 4
xq, xk = torch.randn(1, H, 3, DH), torch.randn(1, HKV, 3, DH)
cos, sin = rope_tables(torch.arange(3), DH)
```

# Starter
```python
def rotate_half(x):
    return TODO


def apply_rope(x, cos, sin):
    return TODO
```

# Solution
```python
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos[None, None] + rotate_half(x) * sin[None, None]
```

# Checks
```python
check("rotate_half", rotate_half(torch.tensor([[1.0, 2.0, 3.0, 4.0]])),
      want=torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))
check("q shape kept", apply_rope(xq, cos, sin), shape=(1, H, 3, DH))
check("works for kv heads too", apply_rope(xk, cos, sin), shape=(1, HKV, 3, DH))
check("position 0 is the identity", apply_rope(xq, cos, sin)[:, :, 0], want=xq[:, :, 0])
rq, half = apply_rope(xq, cos, sin), DH // 2
check("each (i, i+dh/2) pair keeps its length",
      rq[..., :half].pow(2) + rq[..., half:].pow(2),
      want=xq[..., :half].pow(2) + xq[..., half:].pow(2), atol=1e-4)
# The length check is blind to the direction of rotation, so pin one element
# at a non-zero position against the formula written out.
check("rotates the right way at position 2",
      rq[0, 0, 2, 0],
      want=xq[0, 0, 2, 0] * cos[2, 0] - xq[0, 0, 2, half] * sin[2, 0])
```

# Hints
1. 1b "Ellipsis": `...` reaches the last axis at any rank. 1e: `(T, DH)` already lines up with the right end of `(B, Hany, T, DH)`; blank axes in front make that explicit, with no head count in them.
2. `half = x.shape[-1] // 2`, two slices of the last axis, one `cat`. The minus sign goes on the second half, which moves to the front.

# Takeaway
RoPE rotates the pair `(x[i], x[i+dh/2])` in a plane, so that pair's length
is invariant. Checking each pair is tighter than checking the whole head's
norm, and it catches an interleaved pairing.

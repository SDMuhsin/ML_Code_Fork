---
title: Split a fused qkv
difficulty: Easy
---
# Lesson
## Two rules

1. An **integer** index removes that axis.
2. A **slice** keeps the axis, with a new length.

```python demo
w = torch.arange(32).reshape(4, 8)

print("w[0]    ", S(w[0]), "  int: axis gone")
print("w[0:1]  ", S(w[0:1]), "  slice: axis kept, length 1")
print("w[:, -1]", S(w[:, -1]), "  last column")
```

## Ellipsis

`...` means "every axis I did not mention". Use it to reach the last axis
without knowing the rank.

```python demo
print("(T, 128)        ->", S(torch.zeros(5, 128)[..., :64]))
print("(1, 32, T, 128) ->", S(torch.zeros(1, 32, 5, 128)[..., :64]))
```

## Step slices de-interleave

`t[0::2]` takes every other row. This is how you split a matrix whose rows
alternate between two things.

```python demo
print("even rows, col 0:", w[0::2, 0].tolist())
print("odd rows,  col 0:", w[1::2, 0].tolist())
```

# Task
`qkv` is one token's fused q, k, v output: `H*DH` columns of q, then `HKV*DH`
of k, then `HKV*DH` of v. Slice out the three pieces `q`, `k` and `v`, keeping
the batch axis.

# Given
```python
H, HKV, DH = 4, 2, 2
qkv = torch.arange(H * DH + 2 * HKV * DH).float().reshape(1, -1)
```

# Starter
```python
q = TODO
k = TODO
v = TODO
```

# Solution
```python
nq, nk = H * DH, HKV * DH
q = qkv[:, :nq]
k = qkv[:, nq:nq + nk]
v = qkv[:, nq + nk:]
```

# Checks
```python
check("q", q, shape=(1, H * DH))
check("k", k, shape=(1, HKV * DH))
check("v", v, shape=(1, HKV * DH))
check("v starts after q and k", v[0, 0], want=float(H * DH + HKV * DH))
```

# Hints
1. Slice the last axis with `:` on the batch axis, so the batch axis stays. An integer index would remove it.
2. q is columns `0` to `H*DH`. k starts where q ends and is `HKV*DH` wide, not `H*DH`; v is everything after k.

# Takeaway
`HKV*DH` is not `H*DH`. BioMistral has 32 query heads and 8 key/value heads,
so k and v are a quarter of q's width. Confusing the two is the most common
shape bug in this project.

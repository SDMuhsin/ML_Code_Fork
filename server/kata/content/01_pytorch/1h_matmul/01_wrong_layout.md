---
title: Wrong layout, no transpose
difficulty: Easy
---
# Lesson
## Use `@`

The last two axes are the matrix. Every leading axis is a batch axis, and
batch axes broadcast.

```python demo
B, H, T, DH, S_ = 2, 4, 3, 2, 5
q = torch.randn(B, H, T, DH)
k = torch.randn(B, H, S_, DH)
pr = torch.randn(B, H, T, S_)
v = torch.randn(B, H, S_, DH)

print(f"{S(q)} @ {S(k.transpose(-1, -2))} -> {S(q @ k.transpose(-1, -2))}")
print(f"{S(pr)} @ {S(v)} -> {S(pr @ v)}")
```

Those are the two matmuls of attention, one call each.

Use `transpose(-1, -2)`, not `transpose(2, 3)`. It works at any rank.

## F.linear transposes for you

`F.linear(x, W)` computes `x @ W.T`. Its weight is stored
`(out_features, in_features)`, output rows first.

```python demo
D = 8
x = torch.randn(1, D)
Wq = torch.randn(H * DH, D)

print("F.linear(x, Wq)", S(F.linear(x, Wq)),
      " equals x @ Wq.T:", torch.allclose(F.linear(x, Wq), x @ Wq.T))
```

Every HuggingFace weight uses that layout, and so does every matrix in the
packed format.

Watch the naming. The packed format calls a matrix `K x N` with `K=6144,
N=4096`, meaning 6144 output rows and 4096 input columns. `K` is the output
count, not the contraction length, which is the opposite of the usual GEMM
convention. Say "out" and "in" instead of K and N.

# Task
A checkpoint hands you `Wo_bad` in the wrong layout: `(H*DH, D)` where
`o_proj` wants `(D, H*DH)`. Produce `out`, shape `(1, D)`, from `attn_out`
without transposing the weight. At BioMistral's 4096x4096 that is a
32 MiB copy in bf16, per token.

# Given
```python
H, DH, D = 4, 2, 8
attn_out = torch.randn(1, H * DH)
Wo_good = torch.randn(D, H * DH)
Wo_bad = Wo_good.T.contiguous()
```

# Starter
```python
out = TODO
```

# Solution
```python
out = attn_out @ Wo_bad
```

# Checks
```python
check("out", out, shape=(1, D))
check("matches the proper layout", out, want=F.linear(attn_out, Wo_good), atol=1e-4)
```

# Hints
1. `F.linear` transposes its weight for you. Is that what this layout needs?
2. `attn_out` is `(1, H*DH)` and `Wo_bad` is `(H*DH, D)`. Their inner axes already match.

# Takeaway
`F.linear` transposes; a bare `@` does not. In this project you decide
which one each layout needs many times over.

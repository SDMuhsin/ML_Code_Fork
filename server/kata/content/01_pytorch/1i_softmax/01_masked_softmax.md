---
title: Masked softmax at decode
difficulty: Easy
---
# Lesson
## masked_fill overwrites where the mask is True

```python demo
s = torch.zeros(2, 4)
allowed = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)

print(s.masked_fill(~allowed, float("-inf")))
```

Mind the polarity. You pass the positions to overwrite, and you usually
hold an `allowed` mask, so you write `masked_fill(~allowed, ...)`.

```python demo
print(F.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1))
```

Blocked positions come out exactly 0.0 and the rows still sum to 1.

## Softmax has to subtract the max

```python demo
scores = torch.tensor([1000.0, 1001.0, 1002.0])

print("naive exp/sum:", (scores.exp() / scores.exp().sum()).tolist())
print("F.softmax:    ", F.softmax(scores, -1).tolist())
```

`exp(1002)` overflows. Subtracting the row max first avoids that and does
not change the result, because it cancels out of the ratio:

```text
softmax(s)_i = exp(s_i - m) / sum_j exp(s_j - m)
```

`F.softmax` does this for you. The next drill needs the identity.

## A fully masked row gives nan

```python demo
dead = torch.zeros(1, 3).masked_fill(torch.ones(1, 3, dtype=torch.bool), float("-inf"))
print(F.softmax(dead, -1).tolist())
```

With a correct causal mask every query sees at least itself, so this cannot
happen. It is what you see when your mask is transposed.

The CUDA kernel uses `-1e30` instead of `-inf` for this reason. A fully
masked row then gives a wrong number instead of a nan that spreads into
everything after it.

# Task
`scores` is `(1, H, 1, Skv)`: one query row, because `T == 1` at decode.
Mask the blocked positions (where `allowed` is False), softmax over the key
axis accumulating in fp32, and return bf16 as `probs`, shape `(1, H, 1, Skv)`.

# Given
```python
H, Skv = 4, 6
scores = torch.randn(1, H, 1, Skv, dtype=torch.bfloat16)
allowed = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
```

# Starter
```python
probs = TODO
```

# Solution
```python
probs = F.softmax(scores.masked_fill(~allowed, float("-inf")),
                  dim=-1, dtype=torch.float32).bfloat16()
```

# Checks
```python
check("shape", probs, shape=(1, H, 1, Skv))
check("dtype", probs, dtype=torch.bfloat16)
check("masked entries are zero", probs[..., 4:], want=torch.zeros(1, H, 1, 2))
check("rows sum to 1", probs.float().sum(-1), want=torch.ones(1, H, 1), atol=3e-3)
```

# Hints
1. `masked_fill` overwrites where its mask is True, and you hold the positions that are allowed. `-inf` becomes exactly 0 after softmax.
2. `F.softmax` takes a `dtype=` argument that upcasts inside the operation. Convert the result back with `.bfloat16()`.

# Takeaway
`dtype=torch.float32` inside `F.softmax` upcasts within the operation: the
input is bf16, the sum is accumulated in fp32, and the output is bf16. The
kernel does the same.

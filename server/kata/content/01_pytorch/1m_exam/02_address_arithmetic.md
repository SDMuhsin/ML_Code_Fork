---
title: Address arithmetic
difficulty: Easy
---
# Task
- `off`: the flat offset of `t[1, 2, 3]`, from `t.stride()`.
- `back`: the `(i, j, k)` that flat offset 19 maps to in the same tensor, as a tuple of ints.
- `groups`: how many row groups of 48 cover 4096 rows.

# Given
```python
t = torch.arange(24).reshape(2, 3, 4)
```

# Starter
```python
off = TODO
back = TODO
groups = TODO
```

# Solution
```python
off = 1 * t.stride(0) + 2 * t.stride(1) + 3 * t.stride(2)
back = (19 // 12, (19 % 12) // 4, 19 % 4)
groups = (4096 + 48 - 1) // 48
```

# Checks
```python
check("off", torch.tensor(off), want=23)
check("off reads the right element", t.flatten()[off], want=t[1, 2, 3])
check("back", torch.tensor(back), want=torch.tensor([1, 1, 3]))
check("back round trips", t[back], want=t.flatten()[19])
check("groups", torch.tensor(groups), want=86)
```

# Hints
1. 1d Drill 1: the flat offset is each index times its axis's stride, summed. The way back is `//` and `%`, as in 1d's "Flat index to row and column".
2. The strides here are `(12, 4, 1)`: divide by 12 for `i`, then split the remainder by 4. `groups` must round up, because the last group may be short (1d Drill 2).

# Takeaway
`idx // stride` and `idx % stride` is how a CUDA thread works out which element
it owns. Rounding `groups` down would silently drop the last 16 rows:
85 × 48 is 4080.

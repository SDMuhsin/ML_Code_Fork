---
title: Make three tensors
difficulty: Easy
---
# Lesson
## The three sizes

```python demo
x = torch.randn(2, 3, 4)

print("shape", S(x))
print("ndim ", x.ndim)      # how many axes. the rank.
print("numel", x.numel())   # total numbers. the product of the shape.
```

Axis `0` is outermost. Axis `-1` is innermost, and its elements sit next to
each other in memory.

In this project `-1` is nearly always the axis you want: the hidden axis, the
head axis, the vocabulary axis.

## dtype

A tensor holds one kind of number. Mixing them is a common bug.

```python demo
print(torch.randn(3).dtype)                      # float32, the default
print(torch.tensor([7, 2]).dtype)                # int64 from python ints
print((torch.tensor([7, 2]) > 3).dtype)          # bool from a comparison
print(torch.zeros(3, dtype=torch.uint8).dtype)   # what packed weights arrive as
```

# Task
Make these three, using `torch.zeros`:

- `a` with 24 elements and rank 3
- `b` holding bytes (any shape)
- `c` holding booleans (any shape)

# Starter
```python
a = TODO
b = TODO
c = TODO
```

# Solution
```python
a = torch.zeros(2, 3, 4)
b = torch.zeros(5, dtype=torch.uint8)
c = torch.zeros(5, dtype=torch.bool)
```

# Checks
```python
check("a has 24 elements", torch.tensor(a.numel()), want=24)
check("a is rank 3", torch.tensor(a.ndim), want=3)
check("b is bytes", b, dtype=torch.uint8)
check("c is bools", c, dtype=torch.bool)
```

# Hints
1. Rank is the number of axes, so `a` needs a shape with three numbers whose product is 24.
2. `torch.zeros(..., dtype=...)`. Bytes are `torch.uint8`; booleans are `torch.bool`.

# Takeaway
`numel` is the product of the shape and `ndim` is its length. Packed weights
arrive as `uint8` bytes and masks are `bool`; both come back in 1j.

---
title: Into heads, without a copy
difficulty: Easy
---
# Lesson
## Reshaping relabels, it does not move

A shape is metadata over a flat buffer. Reshaping walks that buffer in
**row-major** order: last axis fastest.

```python demo
h = torch.arange(8).float().reshape(1, 8)
heads = h.view(1, 4, 2)          # (batch, head, head_dim)

print(h.tolist())
print(heads)
print("head 0 took elements 0,1 - last axis fastest")
```

- `reshape` returns a view if it can, copies if it must. Use it by default.
- `view` refuses to copy and raises instead, so it doubles as an assertion.
- `-1` means "work this one out".

## transpose moves nothing either

It permutes the strides. The data stays put, which is why the result is no
longer contiguous.

```python demo
A = torch.arange(6).reshape(2, 3)
At = A.transpose(0, 1)

print("A  contiguous:", A.is_contiguous(), " stride", A.stride())
print("At contiguous:", At.is_contiguous(), " stride", At.stride())

try:
    At.view(6)
except RuntimeError as e:
    print("At.view(6) ->", str(e).split(".")[0])

print("At.reshape(6)", At.reshape(6).tolist(), " (it copied)")
print("A.reshape(6) ", A.reshape(6).tolist(), " different numbers")
```

In the last two lines, `reshape` follows the current logical order, so
reshaping after a transpose gives different numbers in a different order.

# Task
Reshape `kflat` into per-head form `k_heads` of shape `(1, HKV, DH)`, in a way
that raises if a copy would be needed.

# Given
```python
HKV, DH = 2, 2
kflat = torch.arange(HKV * DH).float().reshape(1, HKV * DH)
```

# Starter
```python
k_heads = TODO
```

# Solution
```python
k_heads = kflat.view(1, HKV, DH)
```

# Checks
```python
check("k_heads", k_heads, shape=(1, HKV, DH))
check("no copy happened", k_heads.data_ptr() == kflat.data_ptr(), want=True)
```

# Hints
1. Of `reshape` and `view`, only one refuses to copy.
2. `view` takes the new shape the same way `reshape` does: `(1, HKV, DH)`.

# Takeaway
`reshape` would also have passed here, because `kflat` is contiguous and no
copy was needed. When a copy would be needed, `view` raises instead of
copying silently.

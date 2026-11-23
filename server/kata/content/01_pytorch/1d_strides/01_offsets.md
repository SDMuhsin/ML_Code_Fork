---
title: Offsets from strides
difficulty: Easy
---
# Lesson
## stride

`t.stride()` says, per axis, how many **elements** to step to advance one
along that axis.

```python demo
t = torch.arange(24).reshape(2, 3, 4)

print("shape ", S(t))
print("stride", t.stride())
```

The address of `t[i, j, k]` is

```text
i*stride[0] + j*stride[1] + k*stride[2]
```

```python demo
i, j, k = 1, 2, 3
off = i * t.stride(0) + j * t.stride(1) + k * t.stride(2)

print(f"t[{i},{j},{k}]      = {t[i, j, k].item()}")
print(f"flat[{off}]        = {t.flatten()[off].item()}")
```

Transposing reorders the strides and touches no data. That is why it is free,
and why `view` then refuses.

For `view`, the exact condition is that the axes you are merging must be adjacent in
memory: `stride[i] == stride[i+1] * size[i+1]` across each merged run, with
length-1 axes exempt. "Descending strides" is close but not the
rule. `zeros(4,8)[:, :4]` has descending strides `(8, 1)` and `view` still
refuses, because the rows are no longer adjacent.

```python demo
tt = t.transpose(0, 2)
print("stride", tt.stride(), " contiguous:", tt.is_contiguous())
```

## Flat index to row and column

CUDA kernels use the reverse mapping all the time. A kernel gets one flat
thread id and works out which element it owns.

```text
row = idx // N        col = idx % N        for an (K, N) matrix
idx = row * N + col                        the inverse
```

```python demo
N = 6
m = torch.arange(4 * N).reshape(4, N)

for idx in (0, 7, 23):
    r, c = idx // N, idx % N
    print(f"idx {idx:>2} -> ({r}, {c}) -> {m[r, c].item():>2}   back: {r*N+c}")
```

# Task
- `off`: the flat offset of `g[1, 1, 2]`, computed from `g.stride()`.
  Do not hardcode it.
- `rc`: the `(row, col)` that flat index 23 maps to in a 5x7 matrix
  (`K_, N_ = 5, 7` are given).

# Given
```python
# The shape is randomised, so a hardcoded 18 will not do.
_sh = [(2, 3, 4), (3, 2, 5), (4, 2, 6)][torch.randint(3, (1,)).item()]
g = torch.arange(_sh[0] * _sh[1] * _sh[2]).reshape(*_sh)
K_, N_ = 5, 7
print("this run's shape:", S(g), " stride", g.stride())
```

# Starter
```python
off = TODO
rc = TODO
```

# Solution
```python
off = 1 * g.stride(0) + 1 * g.stride(1) + 2 * g.stride(2)
rc = (23 // N_, 23 % N_)
```

# Checks
```python
check("off", torch.tensor(off), want=1 * g.stride(0) + 1 * g.stride(1) + 2 * g.stride(2))
check("off reads the right element", g.flatten()[off], want=g[1, 1, 2])
check("rc", torch.tensor(rc), want=torch.tensor([3, 2]))
```

# Hints
1. Each index is multiplied by the stride of its own axis, and the products are summed. `g.stride(d)` gives one axis's stride.
2. For `rc`, the row is how many whole rows of length `N_` fit before index 23, and the column is what is left over: `//` and `%`.

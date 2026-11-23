---
title: Line up on the named axes
difficulty: Easy
---
# Lesson
## Four ways to add one

```python demo
v = torch.tensor([1.0, 2.0, 3.0, 4.0])

print("v            ", S(v))
print("v[None]      ", S(v[None]), "   front")
print("v[:, None]   ", S(v[:, None]), "   after axis 0 - a column")
print("v[..., None] ", S(v[..., None]), "   end")
print("v[None, None]", S(v[None, None]), "   two at once")
```

`unsqueeze(k)` is the same as putting `None` at position `k`. Use whichever
reads better.

## Outer products

```python demo
print("v * v                  ", S(v * v), "  elementwise")
print("v[None, :] * v[:, None]", S(v[None, :] * v[:, None]), "  outer product")
```

A row times a column is an outer product. RoPE builds its frequency table
this way; you write it in 1m.

## squeeze removes one, conditionally

```python demo
t = torch.zeros(1, 5, 1)

print("t            ", S(t))
print("t.squeeze(0) ", S(t.squeeze(0)), "  removed")
print("t.squeeze(1) ", S(t.squeeze(1)), "  length 5, did nothing")
print("t.squeeze()  ", S(t.squeeze()), "     removed both")
```

Always pass the axis. Bare `squeeze()` removes every length-1 axis, including
the batch axis when the batch size is 1.

## Insertions renumber everything to their right

```python demo
z = torch.zeros(2, 3, 4, 5)

print("z           ", S(z), "  axis 2 has length", z.shape[2])
print("z[:, None]  ", S(z[:, None]), " the old axis 2 is now axis 3")
print("z.sum(1)    ", S(z.sum(1)), "    the old axis 2 is now axis 1")
```

Every insertion and every reduction renumbers the axes after it. Most
mistakes with blank dimensions come from this.

## Lining up from the right

When two tensors meet, PyTorch lines their shapes up from the right.
Axis by axis: equal is fine, one of them being 1 is fine, a missing axis
counts as 1. Anything else is an error.

So `(K, N)` and `(N,)` already line up, and `(K, N)` and `(K,)` do not.
Module 1f covers this in detail.

# Task
Make `small` line up with `big` on the named axes, using `None` indexing,
`unsqueeze` or `squeeze`. Put each result in `ans` under its letter.

| | `big` | `small` | line up on | |
|---|---|---|---|---|
| a | `(B, H, T, DH)` | `(T, DH)` | `T, DH` | already aligned, say so |
| b | `(B, H, T, DH)` | `(H,)` | `H` | needs work |
| c | `(K, N)` | `(K,)` | `K` | needs work |
| d | `(K, N)` | `(N,)` | `N` | already aligned |
| e | `(T, DH)` | `(1, T, DH)` | one axis too many | remove it |

Two of them are already aligned. Spotting those is part of the drill.

# Given
```python
big = {"a": torch.randn(2, 4, 3, 8), "b": torch.randn(2, 4, 3, 8),
       "c": torch.randn(6, 16), "d": torch.randn(6, 16), "e": torch.randn(3, 8)}
small = {"a": torch.randn(3, 8), "b": torch.randn(4), "c": torch.randn(6),
         "d": torch.randn(16), "e": torch.randn(1, 3, 8)}
```

# Starter
```python
ans = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO}
```

# Solution
```python
ans = {"a": small["a"][None, None],
       "b": small["b"][None, :, None, None],
       "c": small["c"][:, None],
       "d": small["d"][None, :],
       "e": small["e"].squeeze(0)}
```

# Checks
```python
check("a lines up on (T,DH)", (big["a"] * ans["a"])[0, 0], want=big["a"][0, 0] * small["a"])
check("b lines up on H", (big["b"] * ans["b"])[0, 2], want=big["b"][0, 2] * small["b"][2])
check("c lines up on K", (big["c"] * ans["c"])[3], want=big["c"][3] * small["c"][3])
check("d lines up on N", (big["d"] * ans["d"])[:, 3], want=big["d"][:, 3] * small["d"][3])
check("e", big["e"] * ans["e"], shape=(3, 8))
check("e dropped the axis", torch.tensor(ans["e"].ndim), want=2)
check("e did it without reshape", torch.tensor(ans["e"].shape[0]), want=3)
```

# Hints
1. Line the shapes up from the right. Where `small` lacks an axis that `big` has *to the right of* the one you want, add a blank axis there.
2. `b` needs one blank axis in front for `B` and two behind for `T, DH`; `c` needs one behind for `N`. For `e`, `squeeze` the leading axis by number.

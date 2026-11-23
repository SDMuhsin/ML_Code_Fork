---
title: Blank dimension placement
difficulty: Easy
---
# Task
Make `small` line up with `big` on the named axes. `None`, `unsqueeze` or
`squeeze` only.

| | `big` | `small` | line up on |
|---|---|---|---|
| a | `(B, H, T, DH)` | `(T, DH)` | `T, DH` |
| b | `(B, H, T, DH)` | `(H,)` | `H` |
| c | `(K, N)` | `(K,)` | `K` |
| d | `(T, DH)` | `(1, T, DH)` | one axis too many: remove it |

Put each result in the dict `ans` under its key.

# Given
```python
big = {"a": torch.randn(2, 4, 3, 8), "b": torch.randn(2, 4, 3, 8),
       "c": torch.randn(6, 16), "d": torch.randn(3, 8)}
small = {"a": torch.randn(3, 8), "b": torch.randn(4),
         "c": torch.randn(6), "d": torch.randn(1, 3, 8)}
```

# Starter
```python
ans = {"a": TODO, "b": TODO, "c": TODO, "d": TODO}
```

# Solution
```python
ans = {"a": small["a"][None, None], "b": small["b"][None, :, None, None],
       "c": small["c"][:, None], "d": small["d"].squeeze(0)}
```

# Checks
```python
check("a on (T,DH)", (big["a"] * ans["a"])[0, 0], want=big["a"][0, 0] * small["a"])
check("b on H", (big["b"] * ans["b"])[0, 2], want=big["b"][0, 2] * small["b"][2])
check("c on K", (big["c"] * ans["c"])[3], want=big["c"][3] * small["c"][3])
check("d", big["d"] * ans["d"], shape=(3, 8))
```

# Hints
1. 1e Drill 1: shapes line up from the right, so a small tensor only needs blank axes where the big one has axes it does not.
2. `b` needs one blank axis in front of `H` and two after it; `c` needs one after `K`. For `d`, pass `squeeze` the axis number.

# Takeaway
Shapes align from the right, so a vector meant for any axis but the last needs
explicit blank axes after it. Always give `squeeze` its axis: a bare
`squeeze()` also removes a batch axis of length 1.

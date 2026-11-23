---
title: Predict reduction shapes
difficulty: Easy
kind: predict
---
# Task
`z` has shape `(2, 3, 4, 5)`. Predict the shape of each expression.

# Given
```python
z = torch.zeros(2, 3, 4, 5)
```

# Items
```json
[
  {"key": "a", "expr": "z.sum(1)", "answer": "(2, 4, 5)"},
  {"key": "b", "expr": "z.sum(2, keepdim=True)", "answer": "(2, 3, 1, 5)"},
  {"key": "c", "expr": "z.mean(0)", "answer": "(3, 4, 5)"},
  {"key": "d", "expr": "z.amax(-1, keepdim=True)", "answer": "(2, 3, 4, 1)"},
  {"key": "e", "expr": "z.argmax(-1)", "answer": "(2, 3, 4)"}
]
```

# Hints
1. A reduction over `dim` removes that axis. With `keepdim=True` the axis stays, with length 1.
2. `argmax` collapses its axis the same way `amax` does, but returns indices instead of values.

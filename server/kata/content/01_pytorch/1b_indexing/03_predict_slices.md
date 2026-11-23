---
title: "Predict: slice shapes"
difficulty: Easy
kind: predict
---
# Task
Write each shape as a tuple before you check.

# Given
```python
w = torch.arange(32).reshape(4, 8)
```

# Items
```json
[
  {"key": "a", "expr": "w[2]", "answer": "(8,)"},
  {"key": "b", "expr": "w[2:3]", "answer": "(1, 8)"},
  {"key": "c", "expr": "w[:, 2:5]", "answer": "(4, 3)"},
  {"key": "d", "expr": "w[0::2]", "answer": "(2, 8)"},
  {"key": "e", "expr": "torch.zeros(1, 32, 128)[..., :64]", "answer": "(1, 32, 64)"}
]
```

# Hints
1. An integer index removes its axis; a slice keeps it with a new length.
2. `...` stands for every axis you did not mention, so `[..., :64]` touches only the last one.

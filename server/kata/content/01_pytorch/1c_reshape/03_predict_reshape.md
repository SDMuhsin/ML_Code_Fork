---
title: "Predict: reshape and transpose"
difficulty: Easy
kind: predict
---
# Lesson
## One more use of view

With a **dtype** argument, `view` reinterprets the bytes instead. It does not
convert values.

```python demo
raw = torch.tensor([0x00, 0x3C, 0x00, 0x40], dtype=torch.uint8)

print("as bytes      ", raw.tolist())
print("view(float16) ", raw.view(torch.float16).tolist(), " 2 values, same bits")
print("to(float16)   ", raw.to(torch.float16).tolist(), " 4 values, converted")
```

`0x3C00` is half-precision 1.0. Module 1j uses this to read the scale factors
out of a packed weight file.

# Task
Write the shape of each expression.

# Given
```python
z = torch.arange(24).reshape(2, 3, 4)
```

# Items
```json
[
  {"key": "a", "expr": "torch.zeros(4, 8).transpose(0, 1).reshape(2, 16)", "answer": "(2, 16)"},
  {"key": "b", "expr": "torch.zeros(2, 3, 8).view(2, 3, 4, 2)", "answer": "(2, 3, 4, 2)"},
  {"key": "c", "expr": "torch.zeros(2, 3, 8).view(2, 3, 4, 2).transpose(1, 2)", "answer": "(2, 4, 3, 2)"},
  {"key": "d", "expr": "z.permute(2, 0, 1).reshape(6, 4)", "answer": "(6, 4)"},
  {"key": "e", "expr": "z.reshape(6, 4)", "answer": "(6, 4)"}
]
```

# Hints
1. `reshape` and `view` give exactly the shape you ask for; `transpose` and `permute` reorder the axes you already have.
2. `permute(2, 0, 1)` puts old axis 2 first, then 0, then 1.

# Takeaway
```python
print("d first row:", z.permute(2, 0, 1).reshape(6, 4)[0].tolist())   # [0, 4, 8, 12]
print("e first row:", z.reshape(6, 4)[0].tolist())                     # [0, 1, 2, 3]
```

Both are `(6, 4)`, but the numbers differ.

---
title: Predict
difficulty: Medium
kind: predict
---
# Task
Write every shape as a tuple. Reason it out; do not run it first.

# Items
```json
[
  {"key": "a", "expr": "torch.randn(2, 3, 4).sum(-1, keepdim=True)", "answer": "(2, 3, 1)"},
  {"key": "b", "expr": "torch.randn(1, 32, 1, 128) * torch.randn(1, 1, 128)", "answer": "(1, 32, 1, 128)"},
  {"key": "c", "expr": "torch.randn(4, 8).transpose(0, 1).reshape(2, 16)", "answer": "(2, 16)"},
  {"key": "d", "expr": "torch.randn(2, 3, 8).view(2, 3, 4, 2).transpose(1, 2)", "answer": "(2, 4, 3, 2)"},
  {"key": "e", "expr": "torch.randn(1, 8, 5, 16) @ torch.randn(1, 8, 16, 3)", "answer": "(1, 8, 5, 3)"},
  {"key": "f", "expr": "torch.randn(5, 1, 7).expand(5, 6, 7)", "answer": "(5, 6, 7)"},
  {"key": "g", "expr": "torch.randn(3)[:, None] * torch.randn(4)[None, :]", "answer": "(3, 4)"},
  {"key": "h", "expr": "torch.randn(2, 8)[:, None, :].expand(2, 3, 8).reshape(2, 24)", "answer": "(2, 24)"},
  {"key": "i", "expr": "torch.randn(10, 4)[torch.tensor([[1, 2], [3, 4]])]", "answer": "(2, 2, 4)"},
  {"key": "j", "expr": "torch.zeros(6, 4, dtype=torch.uint8)[:, :, None] >> torch.arange(8)", "answer": "(6, 4, 8)"},
  {"key": "k", "expr": "torch.zeros(4, 6)[:, 0::2]", "answer": "(4, 3)"},
  {"key": "l", "expr": "torch.randn(2, 4, 3, 8)[..., :4]", "answer": "(2, 4, 3, 4)"}
]
```

# Hints
1. Each item is one rule from 1a–1j: `keepdim` (1g), right-aligned broadcasting (1f), `reshape`/`view`/`transpose` (1c), `@` on the last two axes (1h), `expand` and blank axes (1e), slicing and `...` (1b).
2. Item `i`: indexing an axis with an integer tensor replaces that axis with the index tensor's shape. Item `j` is 1j's bitmap unpack: `(6, 4, 1)` meets `(8,)` from the right.

# Takeaway
Every silent bug in this lesson starts as a shape you assumed instead of
checked. Item `a` is the `keepdim` habit that keeps a reduction lined up
against its input; item `h` is the expand-then-reshape that GQA uses.

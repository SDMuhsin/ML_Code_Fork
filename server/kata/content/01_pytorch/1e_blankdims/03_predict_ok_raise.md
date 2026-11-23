---
title: "Predict: ok or raise"
difficulty: Easy
kind: predict
---
# Task
For each expression, answer `ok` or `raise`. Three of them raise.

# Items
```json
[
  {"key": "a", "expr": "torch.zeros(4, 3) * torch.zeros(3)", "kind": "raises", "answer": "ok"},
  {"key": "b", "expr": "torch.zeros(4, 3) * torch.zeros(4)", "kind": "raises", "answer": "raise"},
  {"key": "c", "expr": "torch.zeros(4, 1).expand(4, 7)", "kind": "raises", "answer": "ok"},
  {"key": "d", "expr": "torch.zeros(4, 2).expand(4, 7)", "kind": "raises", "answer": "raise"},
  {"key": "e", "expr": "torch.zeros(1, 5, 1).squeeze(1)", "kind": "raises", "answer": "ok"},
  {"key": "f", "expr": "torch.zeros(4, 1).expand(4, 7).mul_(2.0)", "kind": "raises", "answer": "raise"}
]
```

# Hints
1. Line shapes up from the right: equal or 1 is fine, anything else raises. `expand` can only grow a length-1 axis.
2. `squeeze(k)` on an axis that is not length 1 does nothing rather than raising. An in-place op on an expanded view would write one address many times.

# Takeaway
`expand` cannot grow a length-2 axis, and you cannot write into a stride-0
view.

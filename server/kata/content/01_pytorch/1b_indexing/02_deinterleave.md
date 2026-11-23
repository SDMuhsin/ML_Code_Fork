---
title: De-interleave gate and up
difficulty: Medium
---
# Task
`gu` is a fused gate+up weight matrix stored interleaved: row `2i` is gate
row `i`, row `2i+1` is up row `i`. Split it back out into `gate` and `up`.

Then set `is_view` by testing in code, not by typing `True`. Does your `gate`
share memory with `gu`?

A literal `True` would also pass that, so `is_view_2` asks about a case with
the opposite answer. A step slice is a view; `clone()` makes a copy. Set
`is_view_2`, again by testing in code. Does `gate.clone()` share memory with `gu`?

# Given
```python
I, D = 16, 8
gu = torch.arange(2 * I * D).float().reshape(2 * I, D)
```

# Starter
```python
gate = TODO
up = TODO
is_view = TODO
is_view_2 = TODO        # does gate.clone() share memory with gu?
```

# Solution
```python
gate = gu[0::2]
up = gu[1::2]
is_view = gate.data_ptr() == gu.data_ptr()
is_view_2 = gate.clone().data_ptr() == gu.data_ptr()
```

# Checks
```python
check("gate", gate, shape=(I, D))
check("up", up, shape=(I, D))
check("gate row 0 is gu row 0", gate[0, 0], want=0.0)
check("up row 0 is gu row 1", up[0, 0], want=float(D))
check("is_view", is_view, want=True)
check("is_view_2", is_view_2, want=False)
```

# Hints
1. A step slice `start::step` on the row axis takes every other row. Which start gives the gate rows, which the up rows?
2. `t.data_ptr()` is the address of a tensor's first element. Two tensors share memory here if those addresses are equal.

# Takeaway
Slices are views. They share storage, so writing through one changes the
other. Call `clone()` when you need a copy.

---
title: nn.Linear by hand
difficulty: Easy
inspect: [y_hand, count_params]
---
# Lesson
## What is inside a layer?

In `2a` you tracked two numbers, `w` and `b`, by hand. A network has thousands.
PyTorch groups them into **layers**. The basic one is `nn.Linear(in, out)`. It
maps each input row of `in` numbers to `out` numbers with

    y = x @ W.T + b

`W` is the layer's `weight`, shape `(out, in)`. `b` is its `bias`, shape
`(out,)`. Both are created with `requires_grad=True` and filled with small
random values. The numbers a training run changes are called the
**parameters**; for a layer that is `W` and `b`.

Each row of `x` is one input (one **sample**). `nn.Linear(in, out)` reads the
last dimension as the `in` numbers of that sample. So 20 inputs of one number
each must be shaped `(20, 1)`, not `(20,)`; `x[:, None]` adds that column.

The demo uses `S(t)`, the Lesson 1 helper for `tuple(t.shape)`.

```python demo
lin = nn.Linear(3, 2)
x = torch.randn(5, 3)
print("S(weight) =", S(lin.weight), "  S(bias) =", S(lin.bias))
print("weight.requires_grad:", lin.weight.requires_grad)
print("lin(x) shape:", S(lin(x)))
print("max |lin(x) - (x @ W.T + b)| =", (lin(x) - (x @ lin.weight.T + lin.bias)).abs().max().item())
```

The weight is stored as `(out, in)`, not `(in, out)`, so the formula has `.T`.
The bias broadcasts across the 5 rows.

`lin.parameters()` yields each parameter tensor of the layer in turn. A
tensor's `numel()` is how many numbers it holds.

# Task
- `y_hand`: the output of the given `lin` on `x`, computed from `lin.weight`
  and `lin.bias` with `@`, without calling `lin(x)`.
- `count_params(module)`: a function returning the total number of numbers in
  `module`'s parameters, as a Python int. It must work for any layer or
  network, not only `lin`.

# Given
```python
lin = nn.Linear(3, 2)
x = torch.randn(5, 3)
```

# Starter
```python
y_hand = TODO

def count_params(module):
    TODO
```

# Solution
```python
y_hand = x @ lin.weight.T + lin.bias

def count_params(module):
    return sum(p.numel() for p in module.parameters())
```

# Checks
```python
check("y_hand", y_hand, want=lin(x), shape=(5, 2))
check("count_params(lin)", count_params(lin), want=8)
check("count_params(nn.Linear(64, 32))", count_params(nn.Linear(64, 32)), want=64 * 32 + 32)
expect("count_params returns an int", isinstance(count_params(lin), int), f"got {type(count_params(lin)).__name__}; numel() is an int, and so is a sum of ints")
```

# Hints
1. Look at the shapes: `x` is `(5, 3)` and `W` is `(2, 3)`. Which way round must `W` be to multiply `x` from the right?
2. Loop over `module.parameters()` and add up.

# Takeaway
`nn.Linear(in, out)` holds `out * in + out` numbers. `nn.Linear(3, 2)` has 8,
`nn.Linear(64, 32)` has 2080. The weight's `(out, in)` layout is why you see
`x @ W.T` in PyTorch code. `torch.nn.functional.linear(x, W, b)` (usually
imported as `F`) computes the same thing without a layer object.

---
title: What each operation saves
difficulty: Medium
inspect: [n_xw, n_vw, n_add, n_exp, n_sum, n_mm, n_mm_both, n_mlp]
---
# Lesson
## Which forward values does a backward pass need?

Each operation in the graph keeps the values its derivative formula needs.
Four examples:

- `y = a * b`: `dy/da = b` and `dy/db = a`. The multiplication saves `b` if
  `a` needs a gradient, and `a` if `b` does.
- `y = tanh(z)`: `dy/dz = 1 - y**2`. It saves its own output `y`.
- `y = a + b`: `dy/da = dy/db = 1`. Nothing to save.
- `a.sum()`, `a.mean()`, `a.T` (transpose): the derivative needs only the
  shape of `a`, not its values. Nothing to save.

`count_saved(fn)` in Shared counts them. PyTorch lets you register a function
that is called each time autograd saves a tensor
(`torch.autograd.graph.saved_tensors_hooks`); the helper registers one that
counts. The demo runs it on
`tanh(w*x + b).sum()` and prints the shape of each saved tensor:

```python demo
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
x = torch.tensor([0.5, -1.0, 2.0])
def show(t):
    print("  saved a tensor of shape", S(t))
    return t
with torch.autograd.graph.saved_tensors_hooks(show, lambda t: t):
    L = torch.tanh(w * x + b).sum()
print("count_saved:", count_saved(lambda: torch.tanh(w * x + b).sum())[0])
```

The two saves are `x` (shape `(3,)`, for `dL/dw`, which needs `x`) and the
tanh output (also `(3,)`). The scalar `w` is not saved because `x` does not
need a gradient. The add and the sum saved nothing.

A Linear layer computes `y = h @ W.T + b`. Its weight gradient is
`g.T @ h`, where `g` is the gradient arriving at `y`, so it needs the input
`h`. The derivative with respect to `h` is `g @ W`, so it needs the weight,
but only if `h` needs a gradient. In the first layer `h` is the input `x`,
which does not.

# Task
For each expression, write how many tensors you think `count_saved` reports,
as an integer, before you run anything. The checks then measure each one and
tell you the true count and why. Names come from Given; `x` and `h` do not
need gradients, `w`, `v`, `W`, `hg` do.

| name | expression |
|---|---|
| `n_xw` | `x * w` |
| `n_vw` | `v * w` |
| `n_add` | `w + v` |
| `n_exp` | `torch.exp(w)` (`d exp(w)/dw = exp(w)`) |
| `n_sum` | `w.sum()` |
| `n_mm` | `h @ W.T` |
| `n_mm_both` | `hg @ W.T` |
| `n_mlp` | `(net(x0) ** 2).mean()` for the reference network `demo_net()`: 3 hidden layers of width 32, 4 Linear layers, 3 tanh |

For `n_mlp`, add up layer by layer from the rules above. The loss `u ** 2`
saves `u`.

# Given
```python
x = torch.randn(5)
w = torch.randn(5, requires_grad=True)
v = torch.randn(5, requires_grad=True)
h = torch.randn(4, 3)
hg = torch.randn(4, 3, requires_grad=True)
W = torch.randn(2, 3, requires_grad=True)
net = demo_net()
x0 = torch.linspace(0, 1, 1000)[:, None]
```

# Starter
```python
n_xw = TODO
n_vw = TODO
n_add = TODO
n_exp = TODO
n_sum = TODO
n_mm = TODO
n_mm_both = TODO
n_mlp = TODO
```

# Solution
```python
n_xw = 1
n_vw = 2
n_add = 0
n_exp = 1
n_sum = 0
n_mm = 1
n_mm_both = 2
n_mlp = 11
```

# Checks
```python
cases = [
    ("n_xw", n_xw, lambda: x * w, "the derivative with respect to w is x; x needs no gradient, so w is not needed"),
    ("n_vw", n_vw, lambda: v * w, "both need gradients: d/dv needs w and d/dw needs v"),
    ("n_add", n_add, lambda: w + v, "the derivative of a sum is 1 for each input; nothing to keep"),
    ("n_exp", n_exp, lambda: torch.exp(w), "the derivative is exp(w), the output itself, so the output is kept"),
    ("n_sum", n_sum, lambda: w.sum(), "the derivative is 1 for every entry; only the shape is kept, which is not a tensor"),
    ("n_mm", n_mm, lambda: h @ W.T, "the weight gradient g.T @ h needs h; h needs no gradient, so W is not kept"),
    ("n_mm_both", n_mm_both, lambda: hg @ W.T, "the weight gradient needs hg and the derivative for hg needs W"),
    ("n_mlp", n_mlp, lambda: (net(x0) ** 2).mean(), "first Linear 1 (its input only), three more Linears 2 each, three tanh outputs, and u for u ** 2: 1 + 6 + 3 + 1"),
]
for name, guess, fn, why in cases:
    true = count_saved(fn)[0]
    expect(f"{name} = {true}", not is_blank(guess) and guess == true, f"{name} is {guess}, the count is {true}: {why}")
```

# Hints
1. For each operation, write its derivative with respect to each input that needs a gradient. Count the tensors that appear in those formulas.
2. For the MLP: count the first Linear, the other three Linears, the tanh layers and the loss separately.

# Takeaway
An operation saves what its derivative needs: `mul` the other operand, `tanh`
and `exp` their output, `add` and `sum` nothing, a Linear its input (and its
weight if the input needs a gradient). These are the choices PyTorch 2.x
makes for these operations, not laws: each backward is written by hand, and
its author picks what to keep. `sin`, SiLU and GELU keep their input rather
than their output, mixed precision (`torch.autocast`) keeps float16 copies,
and compiled code may recompute a value instead of keeping it. When it
matters, measure with `saved_tensors_hooks`, as `count_saved` does. 6 of the MLP's 11 saves are
`(1000, 32)` tensors (per hidden layer one from tanh and one from the next
Linear). They are 3 distinct tensors, since tanh and the next Linear save the
same one, and they hold nearly all the bytes, which the next problem measures.

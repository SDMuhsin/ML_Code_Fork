---
title: First gradient
difficulty: Easy
inspect: [w, b, L, gw, gb, hand_grads]
---
# Lesson
## How much does the loss change if I nudge w?

A model makes a prediction from an input. Here the model is `w * x`: for the
input `x = 3` it predicts `w * 3`, and the right answer (the **target**) is
`y = 7`. The **loss** `(w*x - y)**2` is the squared miss. It is 0 only when
the prediction is exact. Training means changing `w` so that the loss gets
smaller.

To know which way to change `w`, you need the **gradient** of the loss `L`
with respect to `w`: how much `L` changes per unit change in `w`, written
dL/dw. If it is negative, increasing `w` lowers `L`.

PyTorch computes these for you. You mark a tensor with `requires_grad=True`,
compute `L` from it, and call `L.backward()`. Afterwards `w.grad` holds dL/dw.

```python demo
w = torch.tensor(2.0, requires_grad=True)
x, y = 3.0, 7.0
L = (w * x - y) ** 2
L.backward()
print("L      =", L.item())
print("w.grad =", w.grad.item())
print("by hand: 2*(w*x - y)*x =", 2 * (w.item() * x - y) * x)
```

`L.backward()` needs `L` to be a single number. `.item()` turns a one-number
tensor into a Python float.

## Where the gradient comes from

While you compute `L`, PyTorch records each operation, together with the
values its derivative will need (module `2f` shows which). This
record is the **computational graph**. Each result points to the operation that
made it through its `grad_fn` attribute. `backward()` walks this record from
`L` back to `w`, applying the chain rule at each step.

The helper below follows the `grad_fn` links and prints the operations. You do
not need to write such code; read the output.

```python demo
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
x = torch.tensor([0.5, -1.0, 2.0])
y = torch.tanh(w * x + b).sum()

f, chain = y.grad_fn, []
while f is not None:
    chain.append(type(f).__name__)
    nxt = [n for n, _ in f.next_functions if n is not None]
    f = nxt[0] if nxt else None
print(" -> ".join(chain))
```

`torch.tanh` is a smooth S-shaped function with outputs between -1 and 1.
The chain reads from the end back to the start: `sum`, `tanh`, `+`, `*`. The
last node, `AccumulateGrad`, is the one that writes the result into `w.grad`.
Problem `2a-2` shows why it is called "accumulate".

# Task
Given `x`, `y` (Python floats) and the starting values `w = 2.0`, `b = 0.5`:

- make `w` and `b` tensors that autograd tracks,
- compute `L = (w*x + b - y)**2` and call `backward()`,
- `gw`, `gb`: the gradients autograd stored, as Python floats,
- `hand_grads(w, b, x, y)`: a function of four Python floats that returns the
  pair `(dL/dw, dL/db)` from chain-rule formulas, without autograd. The checks
  call it at other values too.

# Given
```python
x, y = 3.0, 7.0
```

# Starter
```python
w = torch.tensor(2.0)   # TODO: let autograd track w
b = torch.tensor(0.5)   # and b
L = (w * x + b - y) ** 2
# TODO: fill w.grad and b.grad

gw = TODO
gb = TODO

def hand_grads(w, b, x, y):
    TODO
```

# Solution
```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
L = (w * x + b - y) ** 2
L.backward()

gw = w.grad.item()
gb = b.grad.item()

def hand_grads(w, b, x, y):
    r = w * x + b - y
    return 2 * r * x, 2 * r
```

# Checks
```python
expect("w is tracked", w.requires_grad, "w.requires_grad is False: create it with requires_grad=True")
expect("b is tracked", b.requires_grad, "b.requires_grad is False: create it with requires_grad=True")
expect("w.grad was filled", w.grad is not None, "w.grad is None: call L.backward() after computing L from the tracked w")
check("gw (autograd)", gw, want=-3.0)
check("gb (autograd)", gb, want=-1.0)
check("hand_grads(2.0, 0.5, 3.0, 7.0)", torch.tensor(hand_grads(2.0, 0.5, 3.0, 7.0)), want=torch.tensor([-3.0, -1.0]))
check("hand_grads(1.0, 0.0, 2.0, 5.0)", torch.tensor(hand_grads(1.0, 0.0, 2.0, 5.0)), want=torch.tensor([-12.0, -6.0]))
check("hand_grads(-1.0, 2.0, 0.5, 0.0)", torch.tensor(hand_grads(-1.0, 2.0, 0.5, 0.0)), want=torch.tensor([1.5, 3.0]))
```

# Hints
1. Name the inside `r = w*x + b - y`, so `L = r**2`. What is dL/dr, and what is dr/dw? dr/db?
2. After `backward()`, the numbers are in `w.grad` and `b.grad`; both are one-number tensors.

# Takeaway
`backward()` gave the same numbers as your formulas: it applies the chain
rule once per recorded operation, from `L` back to `w` and `b`. The
factor `2*(w*x + b - y) = -1.0` appears in both gradients: autograd computes it
once on the way back and reuses it for `w` and for `b`.

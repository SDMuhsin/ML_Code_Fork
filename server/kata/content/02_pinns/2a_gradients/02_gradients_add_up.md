---
title: Gradients add up
difficulty: Medium
inspect: [w, b, losses]
---
# Lesson
## Why does w.grad change when w did not?

Call `backward()` three times on the same function at the same `w`:

```python demo
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
x = torch.tensor([0.5, -1.0, 2.0])
fresh = torch.autograd.grad(torch.tanh(w * x + b).sum(), w)[0]
print("one gradient on its own:", round(fresh.item(), 4))
for i in range(3):
    torch.tanh(w * x + b).sum().backward()
    print("after call", i + 1, " w.grad =", round(w.grad.item(), 4))
```

(`torch.autograd.grad` returns a gradient without touching `.grad`; module
`2d` uses it for every input derivative.) The gradient is 0.0612 each time, but `w.grad` shows
0.0612, 0.1225, 0.1837. `backward()` **adds** the new gradient to whatever
`.grad` already holds. So before each new gradient you set `w.grad = None`.
(`w.grad.zero_()` also works, but only once `.grad` exists. Before the first
`backward()` it is `None`.)

## Gradient descent

To lower the loss, move each number a small step against its gradient:

    w_new = w - lr * dL/dw

`lr` is the **learning rate**, the size of that step. Repeating this is
**gradient descent**. Each repetition is one **step**.

A tensor you create yourself with `requires_grad=True` is a **leaf** of the
graph. A tensor computed from it, like `w * x`, is not. `backward()` stores
gradients only in leaves.

The update itself must not be recorded in the graph: it is not part of the
loss. Wrap it in `with torch.no_grad():`, which turns recording off. Without it,
PyTorch refuses to change a leaf in place:

```python demo
w = torch.tensor(1.0, requires_grad=True)
L = (w * 3.0 - 7.0) ** 2
L.backward()
try:
    w -= 0.1 * w.grad
except RuntimeError as e:
    print("without no_grad:", str(e))
with torch.no_grad():
    w -= 0.1 * w.grad
print("with no_grad: w =", round(w.item(), 4), " still a leaf:", w.is_leaf)
```

`w -= ...` changes `w` in place, so it is still the same leaf tensor and the
next `backward()` fills its `.grad` again. Writing `w = w - ...` without
`no_grad` would make `w` a computed tensor with no `.grad` of its own.

# Task
Fit a line `w*x + b` to the points `(x, y)` in Given by gradient descent.

- Start from `w = 0.0`, `b = 0.0` (tracked by autograd).
- Each of the `steps = 50` steps: compute the loss `L = ((w*x + b - y)**2).mean()`,
  append `L.item()` to the list `losses`, clear the old gradients, call
  `backward()`, then update `w` and `b` with learning rate `lr` under `no_grad`.

The points lie on the line `3x + 1`, so `w` should end near 3 and `b` near 1.

# Given
```python
x = torch.linspace(-2, 2, 20)
y = 3 * x + 1
lr = 0.1
steps = 50
```

# Starter
```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
losses = []
for step in range(steps):
    L = ((w * x + b - y) ** 2).mean()
    losses.append(L.item())
    # TODO: clear old gradients, backward, update w and b
```

# Solution
```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
losses = []
for step in range(steps):
    L = ((w * x + b - y) ** 2).mean()
    losses.append(L.item())
    w.grad = None
    b.grad = None
    L.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
```

# Checks
```python
rw = torch.tensor(0.0, requires_grad=True)
rb = torch.tensor(0.0, requires_grad=True)
ref = []
for _ in range(steps):
    rL = ((rw * x + rb - y) ** 2).mean()
    ref.append(rL.item())
    rw.grad = None; rb.grad = None
    rL.backward()
    with torch.no_grad():
        rw -= lr * rw.grad; rb -= lr * rb.grad
expect("w and b are still tracked leaves", w.is_leaf and w.requires_grad and b.is_leaf and b.requires_grad, "w or b is no longer a leaf with requires_grad: update in place (w -= ...) inside torch.no_grad()")
expect("one loss per step", len(losses) == steps, f"len(losses) is {len(losses)}, want {steps}")
check("w near 3", w.item(), want=3.0, atol=0.05)
check("b near 1", b.item(), want=1.0, atol=0.05)
expect("w.grad holds only the last step's gradient", w.grad is not None and abs(w.grad.item()) < 0.01, f"w.grad is {None if w.grad is None else round(w.grad.item(), 4)}; after 50 steps the true gradient is near 0. Was .grad cleared before every backward()?")
check("losses match plain gradient descent", torch.tensor(losses), want=torch.tensor(ref), rtol=1e-3, atol=1e-6)
```

# Hints
1. The order inside the loop matters: clear, then `backward()`, then update.
2. The update must change the existing `w`, not build a new tensor, and must not be recorded. The second demo shows both.

# Takeaway
The loss went from 14.26 at step 0 to 0.0238 at step 10 and 2e-10 after 50
steps. If you forget to clear `.grad`, each step uses the sum of all past
gradients. The loss still moves and no error appears, but the fit is wrong.
Module `2c` replaces the clearing and the update with two calls on a PyTorch
object, but `.grad` still adds up the same way.

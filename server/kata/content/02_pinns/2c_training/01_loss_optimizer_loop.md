---
title: Optimizer and loop
difficulty: Medium
inspect: [train]
---
# Lesson
## The same loop, with PyTorch doing the bookkeeping

In `2a` you cleared `.grad`, called `backward()`, and updated each number under
`no_grad`. With thousands of parameters you hand that work to an
**optimizer**: an object that holds the list of parameters and knows the
update rule. `opt.zero_grad()` clears every `.grad`; `opt.step()` updates every
parameter. `torch.optim.SGD(params, lr)` applies `p -= lr * p.grad`, the rule
from `2a`.

The **loss function** turns the predictions `u` and the targets `y` into one
number. Here it is the **mean squared error**, `((u - y)**2).mean()`, the
average of the squared misses.

The demo runs the `2a` line fit twice, by hand and with `nn.Linear(1, 1)` plus
SGD, both starting from 0. Steps are numbered from 0: the loss at step 0 is
the one computed before any update.

```python demo
x = torch.linspace(-2, 2, 20)[:, None]
y = 3 * x + 1

w = torch.tensor(0.0, requires_grad=True); b = torch.tensor(0.0, requires_grad=True)
hand = []
for _ in range(50):
    L = ((w * x + b - y) ** 2).mean(); hand.append(L.item())
    w.grad = None; b.grad = None; L.backward()
    with torch.no_grad(): w -= 0.1 * w.grad; b -= 0.1 * b.grad

lin = nn.Linear(1, 1)
with torch.no_grad(): lin.weight.fill_(0.0); lin.bias.fill_(0.0)
opt = torch.optim.SGD(lin.parameters(), lr=0.1)
loop = []
for _ in range(50):
    opt.zero_grad()
    L = ((lin(x) - y) ** 2).mean()
    L.backward()
    opt.step()
    loop.append(L.item())
for s in (0, 10, 49):
    print(f"step {s:2d}  by hand {hand[s]:.4g}   with SGD {loop[s]:.4g}")
```

Both runs print the same losses. Every training loop in this lesson has the
same five lines: `zero_grad`, forward (compute `u`), loss, `backward`, `step`.

`x` has shape `(20, 1)` and so does `y`. If `y` were `(20,)`, then
`lin(x) - y` would broadcast to `(20, 20)` and the loss would compare every
prediction with every target.

## Which optimizer, which learning rate

The line fit had two parameters. `MLP()` from `2b-2` has 2209, and plain SGD
is slow on it. **Adam** (`torch.optim.Adam`) is the usual alternative. It
divides a running average of each parameter's gradient by a running average
of that gradient's size. While a gradient keeps its sign, the parameter moves
by about `lr` per step, whatever the size of the gradient; when the sign flips
from step to step, the averages cancel and the step is smaller. The demo fits `sin(2 pi x)` at 64 points with a
network shaped like `MLP()`, 500 steps, three settings (a few seconds):

```python demo
torch.set_num_threads(1)
xs = torch.linspace(0, 1, 64)[:, None]
ys = torch.sin(2 * math.pi * xs)
def run(opt_cls, lr, steps=500):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                        nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    opt = opt_cls(net.parameters(), lr=lr)
    out = []
    for _ in range(steps):
        opt.zero_grad(); loss = ((net(xs) - ys) ** 2).mean(); loss.backward(); opt.step()
        out.append(loss.item())
    return out
for name, cls, lr in (("SGD  0.1 ", torch.optim.SGD, 0.1), ("Adam 1e-2", torch.optim.Adam, 1e-2), ("Adam 1.0 ", torch.optim.Adam, 1.0)):
    c = run(cls, lr)
    print(f"{name}  step 99 {c[99]:.3g}   step 499 {c[499]:.3g}   lowest {min(c):.3g}")
```

SGD is still at 0.03 after 500 steps; Adam at lr 1e-2 reaches about 1e-5.
At lr 1.0 the tanh units **saturate**: their outputs are pinned near -1 or 1,
so the network outputs a constant and the loss stays at 0.49, the loss of
predicting the mean of the sine (about 0) everywhere. Nothing fails and no
`nan` appears; the only sign is a loss that stops moving, so when a curve
goes flat early, check the learning rate first.

The later problems use Adam with lr 1e-2 unless a problem says otherwise.

# Task
`train(net, x, y, steps, lr, opt_cls)`: make an optimizer with
`opt_cls(net.parameters(), lr=lr)`, run `steps` steps of the loop on the mean
squared error of `net(x)` against `y`, and return the list of the `steps` loss
values as Python floats, the loss of step `i` (computed before update `i`) at
index `i`. The checks run it on the line fit above.

# Given
```python
lx = torch.linspace(-2, 2, 20)[:, None]
ly = 3 * lx + 1
```

# Starter
```python
def train(net, x, y, steps, lr, opt_cls):
    TODO
```

# Solution
```python
def train(net, x, y, steps, lr, opt_cls):
    opt = opt_cls(net.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
```

# Checks
```python
lin = nn.Linear(1, 1)
with torch.no_grad(): lin.weight.fill_(0.0); lin.bias.fill_(0.0)
line_losses = train(lin, lx, ly, 50, 0.1, torch.optim.SGD)
expect("train returns one float per step", isinstance(line_losses, list) and len(line_losses) == 50 and all(isinstance(v, float) for v in line_losses), f"got {type(line_losses).__name__} of length {len(line_losses) if hasattr(line_losses, '__len__') else '?'}; want a list of 50 Python floats (use loss.item())")
check("loss at step 0 (before any update)", line_losses[0], want=14.2632, rtol=1e-3)
check("loss at step 10", line_losses[10], want=0.023823, rtol=1e-2)
expect("loss never goes up", all(b <= a + 1e-7 for a, b in zip(line_losses, line_losses[1:])), "the loss went up at some step; is zero_grad called every step?")
expect("final loss < 1e-3", line_losses[-1] < 1e-3, f"final loss {line_losses[-1]:.3g}")
check("lin.weight was trained to about 3", lin.weight.item(), want=3.0, atol=0.05)
```

# Hints
1. Build the optimizer once, before the loop, not inside it.
2. Record `loss.item()`, not `loss`: a float, not a tensor.

# Takeaway
`train` is the same five lines for any network and any optimizer, which is
why later problems pass the network and the optimizer class in. The losses
matched the hand loop of `2a-2`: 14.26 at step 0, 0.0238 at step 10.

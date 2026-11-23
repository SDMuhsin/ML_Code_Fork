---
title: Why activation functions, and an MLP class
difficulty: Medium
inspect: [MLP]
---
# Lesson
## Does stacking two layers buy anything?

Feed one `nn.Linear` into another. The result is `(x @ W1.T + b1) @ W2.T + b2`,
which is again `x @ W.T + b` with `W = W2 @ W1` and `b = b1 @ W2.T + b2`. Two
linear layers are one linear layer, so the output is a straight line in `x`.

A curve is straight where its **second difference** `u[i+1] - 2*u[i] + u[i-1]`
(on evenly spaced points) is 0. The demo prints it for two stacked layers, and
for the same two layers with `tanh` applied between them:

```python demo
l1, l2 = nn.Linear(1, 8), nn.Linear(8, 1)
x = torch.linspace(-1, 1, 5)[:, None]
def second_diff(u):
    u = u.detach()[:, 0]
    return [round(v, 4) for v in (u[2:] - 2 * u[1:-1] + u[:-2]).tolist()]
print("two layers:      ", second_diff(l2(l1(x))))
print("tanh in between: ", second_diff(l2(torch.tanh(l1(x)))))
```

The first line is zero, or within float rounding (about 1e-8). An **activation
function** is a non-linear function applied to every number between two
layers, and it stops the two layers from collapsing into one. Two common ones:

- `tanh`: smooth, S-shaped, outputs in (-1, 1). `nn.Tanh()`.
- **ReLU**: `max(0, z)`. Zero for negative inputs, the input itself otherwise.
  `nn.ReLU()`. Its graph is made of straight pieces.

## A network as a class

A **multilayer perceptron (MLP)** alternates Linear layers and activation functions
and ends with a Linear layer. Its **width** is the size of the hidden layers (the
ones between input and output). Its **depth** here means the number of hidden
layers, which equals the number of activation functions. Depth 3, width 32, one input:

    Linear(1, 32), Tanh, Linear(32, 32), Tanh, Linear(32, 32), Tanh, Linear(32, 1)

In PyTorch a network is a subclass of `nn.Module`. You write two methods.
`__init__` creates the layers and stores them as attributes. `forward(x)`
computes the output; calling `net(x)` runs `forward`. A layer stored as an
attribute is registered: its parameters show up in `net.parameters()`.
`nn.Sequential(a, b, c)` is a module that applies `a`, then `b`, then `c`.

```python demo
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
    def forward(self, x):
        return self.layers(x)
net = Tiny()
print(net)
print("parameters:", sum(p.numel() for p in net.parameters()), "  output shape:", S(net(x)))
```

13 = `(4 + 4) + (4 + 1)`: both Linear layers were found through `self.layers`.

# Task
Fill in `MLP`. Keep the names: the layers go in `self.layers`, an
`nn.Sequential` in the order shown above. `act` is a class such as `nn.Tanh`;
call `act()` to make each activation layer. The output has one column: input
`(N, din)` gives output `(N, 1)`.

Later problems import this class, so make it work for any `din`, `width`,
`depth` and `act`.

# Given
```python
x = torch.linspace(-1, 1, 7)[:, None]
```

# Starter
```python
class MLP(nn.Module):
    def __init__(self, din=1, width=32, depth=3, act=nn.Tanh):
        super().__init__()
        layers = []
        TODO   # depth times: a Linear, then act(); then the last Linear
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        TODO
```

# Solution
```python
class MLP(nn.Module):
    def __init__(self, din=1, width=32, depth=3, act=nn.Tanh):
        super().__init__()
        layers = []
        d = din
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
```

# Checks
```python
net = MLP()
check("MLP()(x) shape", net(x), shape=(7, 1))
same("first weight (width, din)", S(net.layers[0].weight), (32, 1))
same("last weight (1, width)", S(net.layers[-1].weight), (1, 32))
check("parameter count of MLP()", sum(p.numel() for p in net.parameters()), want=2209)
check("MLP(din=2) on (5, 2)", MLP(din=2)(torch.randn(5, 2)), shape=(5, 1))
check("MLP(width=8, depth=2) parameter count", sum(p.numel() for p in MLP(width=8, depth=2).parameters()), want=(8 + 8) + (64 + 8) + (8 + 1))
n_act = sum(isinstance(m, nn.Tanh) for m in net.layers)
expect("depth=3 means three activations", n_act == 3, f"found {n_act} nn.Tanh layers in self.layers, want 3")
xs = torch.linspace(-1, 1, 201, dtype=torch.float64)[:, None]
torch.manual_seed(0)
relu_net = MLP(act=nn.ReLU).double()
u = relu_net(xs).detach()
flat = ((u[2:] - 2 * u[1:-1] + u[:-2]).abs() < 1e-10).double().mean().item()
expect("act=nn.ReLU is used: the output is made of straight pieces", flat > 0.3, f"only {flat:.0%} of points have zero second difference; with ReLU about half or more should (tanh gives about 0%). Is act() used for every activation?")
```

# Hints
1. Build the list in a loop that runs `depth` times. The first Linear takes `din` inputs; every later one takes `width`.
2. `forward` has one line: the Sequential already knows the order.

# Takeaway
`MLP()` has 2209 parameters: `32 + 32` for the first layer, `1024 + 32` for each
of the two middle layers, `32 + 1` for the last. The last check computed the
second difference of a ReLU network at 201 evenly spaced points. It was exactly
0 at 50-75% of the points (the share varies with the random weights): between the corners where some unit switches on or off, a ReLU
network is a straight line. That matters in module `2h`, where a
physics equation needs a second derivative.

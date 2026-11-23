---
title: Activation memory of an MLP
difficulty: Medium
inspect: [predict_bytes]
---
# Lesson
## How many bytes does one training step keep?

The values a network computes at a hidden layer are called its
**activations**. (The activation function of `2b-2` is what produces them;
here the word means the values.) For `N` points they form an `(N, width)` float32 tensor of
`N * width * 4` bytes (4 bytes per float32). This lesson uses that size as a
unit of memory and calls it **A**. For the reference network of `2f` and
`2g`, `demo_net()` (width 32, depth 3) on 1000 points, A = 1000 * 32 * 4 bytes
= 125 KB. KB and MB mean `2**10` and `2**20` bytes.

The previous problem counted 11 saves for `demo_net()` on 1000 points. The
demo adds up their bytes two ways: every save, and every distinct block of
memory:

```python demo
torch.manual_seed(0)
net = demo_net()
x = torch.linspace(0, 1, 1000)[:, None]
A = 1000 * 32 * 4
raw = []
with torch.autograd.graph.saved_tensors_hooks(lambda t: raw.append(t.numel() * 4) or t, lambda t: t):
    loss = (net(x) ** 2).mean()
n, unique = count_saved(lambda: (net(x) ** 2).mean())
print(f"{n} saves, {sum(raw) / 2**20:.3f} MB if each counted, {unique / 2**20:.3f} MB distinct = {unique / A:.2f} A")
```

The two totals differ because each tanh output is saved twice, once by tanh
(for its derivative) and once by the next Linear (its input, for the weight
gradient). It is one tensor in memory, so it costs once. `count_saved`
returns the distinct bytes. Small items make up the rest: the input `x`, the
output `u`, and the weights. The second demo changes the points and the
width:

```python demo
for N, width in [(1000, 32), (2000, 32), (1000, 64)]:
    torch.manual_seed(0)
    netw = demo_net(width=width)
    xs = torch.linspace(0, 1, N)[:, None]
    print(f"N {N:5d}, width {width:3d}: {count_saved(lambda: (netw(xs) ** 2).mean())[1] / 2**20:.3f} MB")
```

# Task
`predict_bytes(N, width, depth)`: the saved bytes of one plain training step
(`(net(x) ** 2).mean()`) of `demo_net(width=width, depth=depth)` on `N`
points. Work the rule out from the two demos and the `2f-1` count: which
distinct `(N, width)` tensors does each hidden layer keep? Return a Python
int. The checks compare it with `count_saved` on two other
networks and accept a 15% gap (the small items).

# Starter
```python
def predict_bytes(N, width, depth):
    TODO
```

# Solution
```python
def predict_bytes(N, width, depth):
    return depth * N * width * 4
```

# Checks
```python
for N, width, depth in [(2000, 64, 4), (4000, 32, 3)]:
    torch.manual_seed(0)
    cnet = demo_net(width=width, depth=depth)
    cx = torch.rand(N, 1)
    measured = count_saved(lambda: (cnet(cx) ** 2).mean())[1]
    guess = predict_bytes(N, width, depth)
    expect(f"predict_bytes({N}, {width}, {depth}) within 15% of {measured} bytes", abs(guess - measured) <= 0.15 * measured, f"predicted {guess}, measured {measured} ({guess / measured:.2f} times). One A is N * width * 4 bytes; how many per hidden layer?")
```

# Hints
1. How many hidden layers does a network of depth `depth` have, and how many bytes is one `(N, width)` float32 tensor?

# Takeaway
A plain step keeps one A per hidden layer: the tanh output, saved by tanh and by
the next Linear, stored once. For `(2000, 64, 4)` and `(4000, 32, 3)` it was
3% below the measurement; the gap is `x`, `u` and the weights. Saved memory grows linearly
in points, width and depth, and does not depend on the device.

All of it is freed by the next backward pass (`2g-3` gives the full rule
for a PINN step). A loop that runs forward passes
without a backward, such as an evaluation loop that collects
`losses.append(loss)`, keeps every step's graph and its saved values alive
through `loss`. On the reference network, 50 such steps keep 50 * 3.1 A,
about 19 MB, where a loop with `backward()` keeps about 0. Evaluate under `torch.no_grad()` (nothing is saved) or store
`loss.item()` (a Python float, no graph).

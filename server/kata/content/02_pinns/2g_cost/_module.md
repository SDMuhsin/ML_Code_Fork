---
id: 2g
title: The cost of a PINN step
minutes: 40
---
# Intro
All of `2g` uses the reference network of `2f`: `demo_net()` (width 32,
depth 3) on 1000 points, with A = 125 KB. A plain step saves 3.1 A. A step
whose loss contains a second derivative, as in the Poisson PINN, saves
26.2 A and does 4 times the matrix-product FLOPs. By the end you can derive
both numbers from the graph and price a step you have not run.

# Shared
```python
import logging, warnings
import torch.nn as nn
logging.getLogger("torch.utils.flop_counter").disabled = True   # silences a "triton not found" notice
from torch.utils.flop_counter import FlopCounterMode
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

def demo_net(din=1, width=32, depth=3):
    # The same layers as your MLP() from 2b-2. Demos cannot see your code, so they use this.
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)

def count_saved(fn):
    # From 2f: (number of saves, distinct bytes saved) while fn() runs.
    store, n = {}, 0
    def pack(t):
        nonlocal n
        n += 1
        s = t.untyped_storage()
        store[s.data_ptr()] = s.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return n, sum(store.values())

def run_net(net, x):
    # net(x) written with plain function calls; FlopCounterMode cannot count
    # torch.autograd.grad through nn.Module calls.
    for layer in net:
        x = F.linear(x, layer.weight, layer.bias) if isinstance(layer, nn.Linear) else torch.tanh(x)
    return x

def pinn_loss(net, x, order):
    # order 0: plain loss mean(u**2). order k: mean((k-th input derivative)**2), each with create_graph.
    if order == 0:
        return (run_net(net, x) ** 2).mean()
    x = x.detach().requires_grad_()
    g = run_net(net, x)
    for _ in range(order):
        g = torch.autograd.grad(g, x, torch.ones_like(g), create_graph=True)[0]
    return (g ** 2).mean()

def flops(fn):
    # Matrix-product FLOPs counted while fn() runs, and fn's result.
    with FlopCounterMode(display=False) as fc:
        out = fn()
    return fc.get_total_flops(), out

def peak_A(fn, A):
    # Peak GPU memory while fn() runs, above the start, in units of A bytes. fn runs once first,
    # unmeasured: the first matrix product on the GPU allocates library buffers that stay.
    fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / A
```

---
title: Where the x3 in memory comes from
difficulty: Medium
inspect: [per_layer_order2, saved_A]
---
# Lesson
## Order 1: three tensors per hidden layer

`2f-2` found one A per hidden layer for a plain step. With `create_graph=True`
the derivative pass is recorded as graph nodes (`2d-3`), and those nodes save
values too. `saved_by_node` walks the graph of an order-1 loss on a network
with **one** hidden layer and prints each node that saves an `(N, width)`
tensor:

```python demo
def saved_by_node(t, shape):
    seen, todo = set(), [t.grad_fn]
    while todo:
        f = todo.pop()
        if f is None or f in seen:
            continue
        seen.add(f)
        todo += [n for n, _ in f.next_functions]
        big = [a[7:] for a in dir(f) if a.startswith("_saved_")
               and isinstance(getattr(f, a, None), torch.Tensor) and S(getattr(f, a)) == shape]
        if big:
            print(f"  {type(f).__name__:22s} saves {big}")
torch.manual_seed(0)
net1 = demo_net(depth=1)
x = torch.linspace(0, 1, 1000)[:, None]
saved_by_node(pinn_loss(net1, x, 1), (1000, 32))
```

There are three distinct tensors:

- `y`, the tanh output, as in a plain step;
- `g_y`, the gradient arriving at the tanh during the derivative pass;
- `g_z = g_y * (1 - y**2)`, the gradient leaving it.

(In the printout `y` appears under two names, `result` and `output`,
because two nodes save the same block of memory; `g_y` appears as
`grad_output` and `g_z` as `self`.)

Order 2 repeats this on all three.

## Order 2, step by step

The `u_xx` pass is a backward pass through the order-1 graph, recorded again.
Follow one hidden layer.

1. Before it starts, the layer holds three full-size tensors: `y`, `g_y`,
   `g_z`. Each was made by a recorded elementwise operation.
2. The pass carries a gradient back to each of the three, through the
   operation that made it. At `g_z = g_y * (1 - y**2)` both factors depend on
   `x`, so by the product rule the derivative has two terms,
   `(g_y)' * (1 - y**2)` and `g_y * (1 - y**2)'`. Each term is a new
   full-size product, recorded, so its inputs are saved.
3. As at order 1, every one of the layer's tensors gains a gradient arriving
   and a gradient leaving, and both are saved, because the weight backward
   that follows walks back through them.

Matrix products behave differently: in `h @ W.T` only `h` depends on `x`,
so each pass adds one product per product (`2g-1`), and FLOPs only double.

The three-tensor pattern is tanh's (and sigmoid's): their derivative is
computed from the output alone, which is already saved. An activation whose
derivative needs its input, such as `sin` or SiLU, also keeps that input, and
its counts per order differ.

# Task
- `per_layer_order2`: from steps 1-3, the number of distinct `(N, width)`
  tensors one hidden layer keeps in an order-2 step (an integer).
- `saved_A(depth, order)`: the saved memory of a step of `demo_net(depth=depth)`
  in units of A, for orders 0, 1, 2, from the per-layer counts (ignore the
  small items).

The checks measure how much one extra hidden layer adds at each order, and
compare `saved_A` with `count_saved` on networks of depth 2 and 5 (12%).

# Starter
```python
per_layer_order2 = TODO

def saved_A(depth, order):
    TODO
```

# Solution
```python
per_layer_order2 = 9

def saved_A(depth, order):
    return depth * [1, 3, per_layer_order2][order]
```

# Checks
```python
A5 = 500 * 32 * 4
cx = torch.rand(500, 1)
def measured_A(depth, order):
    torch.manual_seed(0)
    return count_saved(lambda: pinn_loss(demo_net(depth=depth), cx, order))[1] / A5
per_layer = [measured_A(2, o) - measured_A(1, o) for o in range(3)]
print("one extra hidden layer adds, at orders 0, 1, 2:", [round(v, 2) for v in per_layer], "A")
expect(f"per_layer_order2 = {round(per_layer[2])}", not is_blank(per_layer_order2) and per_layer_order2 == round(per_layer[2]), f"you said {per_layer_order2}, one extra layer adds {per_layer[2]:.2f} A. The layer starts the u_xx pass with 3 tensors; each gains a gradient arriving and one leaving.")
for depth in (2, 5):
    for order in range(3):
        m = measured_A(depth, order)
        p = saved_A(depth, order)
        expect(f"saved_A({depth}, {order}) within 12% of {m:.1f} A", abs(p - m) <= 0.12 * m, f"predicted {p}, measured {m:.1f} A. Per hidden layer: 1 A plain, 3 A at order 1, per_layer_order2 at order 2.")
```

# Hints
1. Start from 3 tensors. How many does each one bring with it after a recorded pass, counting itself?
2. `saved_A` is the depth times a per-layer count that depends on the order.

# Takeaway
Each extra hidden layer adds 1.0, 3.0 and 9.0 A at orders 0, 1 and 2: with
tanh, each recorded pass turns every full-size tensor of a layer into three.
Activations whose derivative needs their input (`sin`, SiLU, GELU) keep that
input too and grow by other factors (measured per layer: sin 2, 5, 12 A;
SiLU 2, 8, 19 A), so count them with `count_saved` before you price such a
network. The reference network saves 3.1, 9.2 and 26.2 A. The x3
holds for orders 1 and 2, the PINN cases; from order 3 on some tensors are
shared and the factor drops below 3.

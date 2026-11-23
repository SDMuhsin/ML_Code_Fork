---
id: 02_pinns
title: Lesson 2 · Physics-informed networks
---
# Intro
This lesson starts from tensors, as Lesson 1 left them, and ends with a
network trained to obey a physics equation: a **physics-informed neural
network** (PINN). On the way you will measure how much memory and compute
each training step needs. Every new term is defined in the problem that first
uses it.

- `2a`-`2c` (about 45 minutes): how a network learns from examples. PyTorch
  computes derivatives for you, layers are built from matrix products, and a
  short loop repeats "measure the error, adjust the numbers".
- `2d`: derivatives of the network's output with respect to its input, such
  as `du/dx`. A physics equation is written with these.
- `2e`: the first PINNs, for two equations with one input each. `2h`: two
  ways such a network goes wrong while its training numbers look fine.
- `2f`, `2g`: what a PINN step keeps in memory and how many multiply-adds it
  does, and why each extra derivative costs more. You will predict both
  before running a step.
- `2i`: an equation with two inputs, space and time, and when a PINN is worth
  using instead of a classic solver.

Each problem prints the numbers it talks about. Run the examples before you
start the task.

# Recall
For review after the modules, not before. A day after finishing, and again a
week later, write each from memory. The module id says where it is taught.

1. `2a`: a loss of `w` and `b`, its `backward()`, and a gradient descent loop that zeroes `.grad` and updates under `torch.no_grad()`
2. `2b`: `x @ W.T + b` for an `nn.Linear`, its parameter count, and the `MLP(din, width, depth, act)` class
3. `2c`: the five-line training loop with an optimizer
4. `2d`: `ddx(u, x)` with `grad_outputs` and `create_graph=True`, and why ones gives one derivative per point
5. `2e`: the residual and the full loss of the decay PINN and of the 1D Poisson PINN
6. `2h`: why a ReLU network cannot fit a residual that contains `u''`, and how to set the boundary weight from the printed boundary loss
7. `2f`: what `mul`, `tanh`, `add` and a Linear layer save for backward, and why
8. `2g`: FLOPs of a matrix product, of a derivative pass and of a weight backward; the totals for derivative orders 0-2
9. `2g`: the saved memory per hidden layer at orders 0-2, and the peak
10. `2g`: sibling passes vs stacked passes, in FLOPs and memory
11. `2i`: the heat residual `u_t - k u_xx` from two `ddx` calls on an `(N, 2)` input
12. `2i`: when a classic solver beats a PINN, and when a PINN is worth it

# Traps
For review after the modules. Each gives a wrong answer with no error
message; the module id says where it is shown.

1. `2a`: `.grad` adds up across `backward()` calls. Zero it every step.
2. `2b`: two `nn.Linear` layers with no activation function between them are one linear map.
3. `2c`: predictions `(N, 1)` minus targets `(N,)` broadcast to `(N, N)`. The loss is computed, and it is wrong.
4. `2c`: a learning rate that is too large often gives no `nan`, only a loss that stops moving.
5. `2d`: `grad_outputs = ones` gives per-point derivatives only if the points do not interact. A batch mean breaks it; `nn.BatchNorm1d` in training mode is such a mean.
6. `2d`: a loss that contains an input derivative needs `create_graph=True`. `retain_graph=True` runs and gives a wrong weight gradient.
7. `2e`: the equation alone has many solutions. Without the initial or boundary term the network finds another one.
8. `2h`: a ReLU network has `u'' = 0`, so a residual with `u''` cannot train.
9. `2h`: a boundary weight that is too small leaves the ends off; one too large leaves the inside unfitted. Print the parts of the loss, not only the total.
10. `2f`: keeping `loss` from steps with no backward keeps every step's graph alive. Keep `loss.item()` instead.
11. `2g`: budget for the peak, not the saved memory. The weight backward adds gradient buffers before it frees the saved values.
12. `2i`: `g[:, 0]` has shape `(N,)`; `g[:, 0:1]` keeps `(N, 1)`. The wrong one broadcasts the residual to `(N, N)`.

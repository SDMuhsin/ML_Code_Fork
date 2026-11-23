---
id: 2l
title: "SAGE: trace, emit, exec"
minutes: 62
---
# Intro
`2k-2` wrote 5 adjoint lines by hand. The Navier-Stokes residual of `2m`
needs 40, and the residual in `pinn-solver-accel` needs 331. A wrong one
gives a wrong gradient and no error. **SAGE** (symbolic adjoint generation)
writes them for you. It runs the residual once on objects that record each
operation. It walks the record backwards, applying the rules of `2k-2`. It
prints the result as Python source and turns that source into a function.
The whole program is under 100 lines. By the end you have
the same pieces as `sage.py` in the real code (the package
`usask_accel/` of `pinn-solver-accel`, read in `2m-2`), in PyTorch.

# Shared
```python
NS_NAMES = ["u", "v", "p", "u_x", "u_y", "v_x", "v_y", "p_x", "p_y", "u_xx", "u_yy", "v_xx", "v_yy"]

def make_ns_loss(nu):
    # Per-point loss of 2D steady incompressible Navier-Stokes (2m-1 explains it): three residuals, squared, added.
    nu = float(nu)
    def loss_point(Fd):
        u, v = Fd["u"], Fd["v"]
        r1 = u * Fd["u_x"] + v * Fd["u_y"] + Fd["p_x"] - nu * (Fd["u_xx"] + Fd["u_yy"])
        r2 = u * Fd["v_x"] + v * Fd["v_y"] + Fd["p_y"] - nu * (Fd["v_xx"] + Fd["v_yy"])
        r3 = Fd["u_x"] + Fd["v_y"]
        return r1 * r1 + r2 * r2 + r3 * r3
    return loss_point

def make_burgers_loss(nu):
    # Per-point loss of 2k-2: l = r * r with r = u u_x - nu u_xx - f.
    nu = float(nu)
    def loss_point(Fd):
        r = Fd["u"] * Fd["u_x"] - nu * Fd["u_xx"] - Fd["f"]
        return r * r
    return loss_point

BURGERS_NAMES = ["u", "u_x", "u_xx", "f"]

def as_dict(stack, names):
    # (N, n) field stack -> {name: (N, 1) column}
    return {n: stack[:, i:i + 1] for i, n in enumerate(names)}

def autograd_adjoint(loss_point, stack, names):
    # Reference: d mean(loss_point) / d stack, by autograd. Fields the loss does not read get 0.
    s = stack.detach().clone().requires_grad_()
    loss_point(as_dict(s, names)).mean().backward()
    return s.grad if s.grad is not None else torch.zeros_like(s)

def forward_line(entry):
    # One line of Python that computes an entry's output from its inputs (used in 2l-3).
    op, out = entry[0], entry[-1]
    if op == "add":
        return f"{out.name} = {entry[1].name} + {entry[2].name}"
    if op == "sub":
        return f"{out.name} = {entry[1].name} - {entry[2].name}"
    if op == "mul":
        return f"{out.name} = {entry[1].name} * {entry[2].name}"
    if op == "smul":
        return f"{out.name} = {entry[1]!r} * {entry[2].name}"
    if op == "cadd":
        return f"{out.name} = {entry[1].name} + {entry[2]!r}"
    raise ValueError(f"unknown op {op}")
```

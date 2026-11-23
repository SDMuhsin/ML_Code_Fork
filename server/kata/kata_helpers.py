"""Helpers every kata problem sees, the same ones the lesson-1 notebooks import.

TODO      a placeholder you replace
S(t)      shape as a tuple
check     check one answer against a shape / dtype / value
same      two shapes must match
expect    a plain condition must hold
is_blank  True if an answer is still the TODO placeholder
DEVICE    the device this run targets ("cpu" or "cuda")
gen(seed) a torch.Generator on DEVICE, so seeded setup works on either device

Unlike the notebook versions, these never raise. Every check is recorded, so a
run shows all of them, not just the first failure.
"""
import torch

DEVICE = "cpu"
_results = []


class _Todo:
    def __repr__(self):
        return "TODO"

    # Arithmetic on a TODO should fail with a message that says what happened.
    def _bad(self, *a, **k):
        raise TypeError("a TODO placeholder is still in your code")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _bad
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = __matmul__ = _bad
    __rmatmul__ = __pow__ = __neg__ = __getitem__ = __call__ = __index__ = _bad
    __lt__ = __le__ = __gt__ = __ge__ = __and__ = __or__ = __invert__ = _bad
    __lshift__ = __rshift__ = __mod__ = __rmod__ = __iter__ = __len__ = _bad
    __float__ = __int__ = __bool__ = _bad

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise TypeError("a TODO placeholder is still in your code")


TODO = _Todo()


def is_blank(x):
    """True if x is still the TODO placeholder."""
    return isinstance(x, _Todo)


def S(t):
    """Shape as a tuple."""
    return tuple(t.shape)


def gen(seed=0):
    """A seeded generator on the current device."""
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _reset(device):
    global DEVICE
    DEVICE = device
    _results.clear()


def _record(label, passed, details=()):
    if len(_results) < 300:
        _results.append({"label": str(label), "passed": bool(passed),
                         "details": [str(d)[:2000] for d in details]})
    return bool(passed)


class printopts:
    """Temporarily change torch's print options (restores them afterwards)."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        from torch._tensor_str import PRINT_OPTS as P
        self.old = dict(precision=P.precision, threshold=P.threshold, edgeitems=P.edgeitems,
                        linewidth=P.linewidth, sci_mode=P.sci_mode)
        torch.set_printoptions(**self.kw)

    def __exit__(self, *exc):
        torch.set_printoptions(**self.old)


def _fmt(t):
    """A compact, readable rendering of a small tensor."""
    if t.numel() == 1:
        v = t.reshape(()).item()
        return f"{v:.6g}" if isinstance(v, float) else repr(v)
    with printopts(precision=4, sci_mode=False, linewidth=100, threshold=40, edgeitems=3):
        s = repr(t)
    s = s.replace("tensor(", "").rstrip(")")
    return " ".join(s.split()) if t.ndim <= 1 else s


def _as_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x).cpu()


def _tname(x):
    return type(x).__name__


def check(label, got, want=None, shape=None, dtype=None, atol=1e-5, rtol=1e-4):
    """Check one answer. Records what went wrong; never raises."""
    bad = []
    if is_blank(got):
        bad.append("not filled in yet: still TODO")
        return _record(label, False, bad)

    if type(got).__module__ == "torch.return_types":
        tn = _tname(got)
        bad.append(f".{tn}(dim=...) returns a (values, indices) pair, not a tensor. "
                   f"Use .values, or the a-prefixed form: a{tn}.")
        return _record(label, False, bad)

    shape_reported = False
    if shape is not None:
        if not torch.is_tensor(got):
            bad.append(f"expected a tensor, got {_tname(got)}: {got!r:.80}")
            shape_reported = True
        elif S(got) != tuple(shape):
            bad.append(f"shape is {S(got)}, want {tuple(shape)}")
            shape_reported = True

    if dtype is not None:
        if torch.is_tensor(got):
            if got.dtype != dtype:
                bad.append(f"dtype is {got.dtype}, want {dtype}")
        elif not shape_reported:
            bad.append(f"expected a tensor, got {_tname(got)}")

    if want is not None:
        try:
            g, w = _as_cpu(got), _as_cpu(want)
        except Exception:
            bad.append(f"expected a tensor or a number, got {_tname(got)}: {got!r:.80}")
            return _record(label, False, bad)
        if S(g) != S(w):
            if not shape_reported:
                bad.append(f"shape is {S(g)}, want {S(w)}")
        else:
            gd, wd = g.double(), w.double()
            if torch.isnan(gd).any() and not torch.isnan(wd).any():
                bad.append("your result contains nan")
            elif not torch.allclose(gd, wd, atol=atol, rtol=rtol):
                diff = (gd - wd).abs()
                diff = torch.nan_to_num(diff, nan=float("inf"))
                if g.numel() == 1:
                    bad.append(f"got {_fmt(g)}, want {_fmt(w)}")
                elif g.numel() <= 24:
                    bad.append(f"got  {_fmt(g)}")
                    bad.append(f"want {_fmt(w)}")
                    bad.append(f"largest gap {diff.max().item():.3e}")
                else:
                    idx = torch.unravel_index(diff.argmax(), diff.shape)
                    at = tuple(int(i) for i in idx)
                    n_bad = int((diff > atol + rtol * wd.abs()).sum())
                    bad.append(f"values differ at {n_bad} of {g.numel()} positions; "
                               f"largest gap {diff.max().item():.3e} at index {list(at)}")
                    bad.append(f"there: got {g[at].item():.6g}, want {w[at].item():.6g}")
    return _record(label, not bad, bad)


def same(label, a, b):
    """Two shapes (or tuples) must match."""
    if is_blank(a):
        return _record(label, False, ["not filled in yet: still TODO"])
    if isinstance(a, int):
        return _record(label, False, [f"{a!r} is an int, not a tuple. A one-axis shape "
                                      f"is written ({a},) with the trailing comma."])
    try:
        ta = tuple(a)
    except TypeError:
        return _record(label, False, [f"expected a tuple like (2, 3), got {a!r}"])
    tb = tuple(b)
    if ta != tb:
        return _record(label, False, [f"you said {ta}, it is {tb}"])
    return _record(label, True, [f"= {tb}"])


def expect(label, cond, why=""):
    """A plain condition that must be true."""
    if is_blank(cond):
        return _record(label, False, ["not filled in yet: still TODO"])
    ok = bool(cond)
    return _record(label, ok, [] if ok else ([why] if why else []))


__all__ = ["TODO", "S", "check", "same", "expect", "is_blank", "gen"]

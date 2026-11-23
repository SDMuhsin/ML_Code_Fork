"""Kata worker. One process runs exactly one job, then exits.

The Node side (pool.js) keeps a few of these started ahead of time, so torch
(and the CUDA context) are already loaded when you press Run. A fresh process
per job means nothing you do in one run can leak into the next.

Protocol, all on stdout, one line each:
    @@KATA_READY@@{env json}      once torch is imported and warm
    <reads one job as a JSON line from stdin>
    @@KATA_RESULT@@{result json}  then exits

Job modes
    run      shared -> uses -> given -> user -> checks, plus a variable inspector
    predict  shared -> given, then evaluate each item against the learner's answer
    demo     shared -> earlier demos (silently) -> this demo, echoing like a notebook
    scratch  shared -> uses -> given -> user (silently) -> scratch, echoing
"""
import ast
import builtins
import contextlib
import io
import json
import linecache
import math
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

READY = "@@KATA_READY@@"
RESULT = "@@KATA_RESULT@@"
REAL_STDOUT = sys.stdout
MAX_TEXT = 60_000


def emit(tag, obj):
    REAL_STDOUT.write(tag + json.dumps(obj, default=str) + "\n")
    REAL_STDOUT.flush()


def env_info(torch):
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "executable": sys.executable,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(device_name=p.name, capability=f"{p.major}.{p.minor}",
                    total_memory=p.total_memory, sm_count=p.multi_processor_count)
    return info


# --------------------------------------------------------------------------- #
# Running source with good errors
# --------------------------------------------------------------------------- #

class Cap:
    """Captures stdout+stderr of a phase into one buffer."""

    def __init__(self):
        self.buf = io.StringIO()

    @contextlib.contextmanager
    def __call__(self):
        with contextlib.redirect_stdout(self.buf), contextlib.redirect_stderr(self.buf):
            yield

    def text(self):
        s = self.buf.getvalue()
        return s if len(s) <= MAX_TEXT else s[:MAX_TEXT] + "\n... (output truncated)"


def register(filename, src):
    lines = [ln + "\n" for ln in src.split("\n")]
    linecache.cache[filename] = (len(src), None, lines, filename)


def error_info(exc, phase, focus=("<solution>",)):
    """Type, message, the line in `focus` files it came from, and a traceback
    trimmed to the frames the learner can do something about."""
    tb = traceback.extract_tb(exc.__traceback__)
    line = None
    for fr in tb:
        if fr.filename in focus:
            line = fr.lineno
    keep = [fr for fr in tb if fr.filename.startswith("<")
            and fr.filename not in ("<string>",)]
    parts = ["Traceback (most recent call last):\n"]
    for fr in keep:
        where = {"<solution>": "your code", "<checks>": "checks", "<given>": "given",
                 "<shared>": "module setup", "<scratch>": "playground"}.get(fr.filename, fr.filename)
        parts.append(f'  {where}, line {fr.lineno}' + (f", in {fr.name}" if fr.name != "<module>" else "") + "\n")
        if fr.line:
            parts.append(f"    {fr.line.strip()}\n")
    msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    col = None
    message = str(exc) or type(exc).__name__
    if isinstance(exc, SyntaxError):
        if exc.filename in focus:
            line, col = exc.lineno, exc.offset
        message = f"{exc.msg} (line {exc.lineno})"
        parts = [f"{type(exc).__name__}: {exc.msg}\n"]
        if exc.text:
            text = exc.text.rstrip("\n")
            parts.append(f"  line {exc.lineno}:  {text.strip()}\n")
            if exc.offset:
                lead = len(text) - len(text.lstrip())
                parts.append(" " * (10 + len(str(exc.lineno)) + max(0, exc.offset - 1 - lead)) + "^\n")
    else:
        parts.append(msg)
    return {"phase": phase, "type": type(exc).__name__, "message": message,
            "line": line, "col": col, "traceback": "".join(parts)}


class _BareTodoInFunctions(ast.NodeTransformer):
    """A bare `TODO` line inside a function body is a no-op in Python, so the
    function would run on and fail later with some unrelated error (an empty
    list passed to torch.stack, a cache that is still zeros). Turn it into
    `TODO()`, which raises "a TODO placeholder is still in your code" at that
    line. Top-level bare TODOs are left alone: raising there would abort the
    whole run before the later definitions exist."""

    def __init__(self):
        self.depth = 0

    def _func(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _func

    def visit_Expr(self, node):
        v = node.value
        if self.depth and isinstance(v, ast.Name) and v.id == "TODO":
            call = ast.Call(func=ast.Name(id="TODO", ctx=ast.Load()), args=[], keywords=[])
            node.value = ast.copy_location(call, v)
            ast.fix_missing_locations(node)
        return node


def compile_src(src, filename):
    if "TODO" not in src:
        return compile(src, filename, "exec")
    tree = ast.parse(src, filename)
    tree = _BareTodoInFunctions().visit(tree)
    return compile(tree, filename, "exec")


def run_src(src, ns, filename, cap, phase, focus=("<solution>",)):
    """Exec src. Returns None, or an error dict."""
    if not src or not src.strip():
        return None
    register(filename, src)
    try:
        code = compile_src(src, filename)
    except SyntaxError as e:
        return error_info(e, phase, focus=(filename,) + tuple(focus))
    try:
        with cap():
            exec(code, ns)
    except BaseException as e:  # noqa: BLE001 - we report everything, incl. SystemExit
        if isinstance(e, KeyboardInterrupt):
            raise
        return error_info(e, phase, focus=(filename,) + tuple(focus) if filename == "<solution>" else focus)
    return None


def run_echo(src, ns, filename, cap, phase):
    """Exec src like a notebook cell: the value of a trailing expression is printed."""
    if not src or not src.strip():
        return None
    register(filename, src)
    try:
        tree = ast.parse(src, filename)
    except SyntaxError as e:
        return error_info(e, phase, focus=(filename,))
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    try:
        with cap():
            exec(compile(tree, filename, "exec"), ns)
            if last is not None:
                val = eval(compile(last, filename, "eval"), ns)
                if val is not None:
                    print(repr(val))
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        return error_info(e, phase, focus=(filename,))
    return None


def todo_lines(src):
    """Line numbers where the name TODO is still used."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "TODO"})


def starter_names(starter):
    """The names a learner is asked to produce: top-level assignment targets and
    function definitions in the starter code (not inside function bodies)."""
    try:
        tree = ast.parse(starter or "")
    except SyntaxError:
        return []
    out = []

    def add(n):
        if n not in out and not n.startswith("_"):
            out.append(n)

    def targets(t):
        if isinstance(t, ast.Name):
            add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                targets(e)

    def walk(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(s.name)
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    targets(t)
            elif isinstance(s, (ast.AugAssign, ast.AnnAssign)):
                targets(s.target)
            elif isinstance(s, (ast.For, ast.While, ast.If, ast.With)):
                walk(s.body)
                walk(getattr(s, "orelse", []))
    walk(tree.body)
    return out


# --------------------------------------------------------------------------- #
# The variable inspector
# --------------------------------------------------------------------------- #

def describe(name, v, torch, H):
    if H.is_blank(v):
        return {"name": name, "kind": "todo", "summary": "TODO (not filled in yet)"}
    if torch.is_tensor(v):
        d = {"name": name, "kind": "tensor", "shape": list(v.shape),
             "dtype": str(v.dtype).replace("torch.", ""), "device": str(v.device),
             "stride": list(v.stride()), "contiguous": v.is_contiguous(), "numel": v.numel()}
        try:
            with H.printopts(precision=4, sci_mode=False, linewidth=88, threshold=60, edgeitems=3):
                d["preview"] = repr(v.detach().cpu())[:3000]
        except Exception as e:  # noqa: BLE001
            d["preview"] = f"<could not print: {e}>"
        return d
    if type(v).__module__ == "torch.return_types":
        return {"name": name, "kind": "pair",
                "summary": f"torch.return_types.{type(v).__name__}: a (values, indices) pair, not a tensor"}
    if callable(v):
        import inspect
        try:
            sig = str(inspect.signature(v))
        except (TypeError, ValueError):
            sig = "(...)"
        return {"name": name, "kind": "function", "summary": f"def {name}{sig}"}
    try:
        r = repr(v)
    except Exception as e:  # noqa: BLE001
        r = f"<repr failed: {e}>"
    return {"name": name, "kind": type(v).__name__, "summary": r[:600]}


# --------------------------------------------------------------------------- #
# Checks: one statement at a time, so one failure does not hide the rest
# --------------------------------------------------------------------------- #

CHECK_FUNCS = {"check", "same", "expect"}


def _check_calls(stmt):
    return [n for n in ast.walk(stmt) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id in CHECK_FUNCS]


def _label_text(node, ns):
    """The text of a check's label argument; f-string holes become '…'."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "…" for v in node.values)
    try:
        return str(eval(compile(ast.Expression(node), "<label>", "eval"), ns))
    except Exception:  # noqa: BLE001
        return None


def stmt_label(stmt, src, ns):
    calls = _check_calls(stmt)
    if calls and calls[0].args:
        text = _label_text(calls[0].args[0], ns)
        if text:
            return text
    seg = (ast.get_source_segment(src, stmt) or "").strip().split("\n")[0]
    return f"check at line {stmt.lineno}: {seg[:70]}"


def _stored_names(stmt):
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _loaded_names(stmt):
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _user_funcs_called(stmt, ns):
    """Functions the learner wrote (in <solution>) that this statement calls."""
    out = []
    for n in ast.walk(stmt):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            f = ns.get(n.func.id)
            code = getattr(f, "__code__", None)
            if code is not None and code.co_filename == "<solution>" and n.func.id not in out:
                out.append(n.func.id)
    return out


def _ticks(names):
    return ", ".join(f"`{n}`" for n in names)


def run_checks(src, ns, cap, H, user_has_todo):
    """Run the checks one top-level statement at a time.

    A statement that calls check/same/expect is a *check*: it always records at
    least one result. Anything else (an assignment, a loop, a print) is *setup*
    for later checks. When setup cannot run because the learner's code is not
    finished, it records nothing; the names it would have defined are marked
    unavailable, and each check that needs them says why. So a learner sees
    their real checks with one clear reason, not a fake "check at line 1"
    followed by a string of NameErrors."""
    register("<checks>", src)
    try:
        tree = ast.parse(src, "<checks>")
    except SyntaxError as e:  # an authoring bug, not the learner's
        H._record("checks failed to parse (authoring error)", False, [str(e)])
        return
    unavailable = {}  # name -> the reason it could not be computed
    for stmt in tree.body:
        is_check = bool(_check_calls(stmt))
        stored = _stored_names(stmt)
        loaded = _loaded_names(stmt)

        # Reading a variable still set to TODO, or one an earlier setup step
        # could not compute: say so by name instead of running the statement.
        blank = sorted(n for n in loaded if H.is_blank(ns.get(n)))
        deps = sorted(n for n in loaded if n in unavailable and n not in stored)
        reason = None
        if blank:
            reason = f"not filled in yet: {_ticks(blank)} {'is' if len(blank) == 1 else 'are'} still TODO"
        elif deps:
            reason = unavailable[deps[0]]
        if reason:
            if is_check:
                H._record(stmt_label(stmt, src, ns), False, [reason])
            else:
                for n in stored:
                    unavailable[n] = reason
            continue

        mod = ast.Module(body=[stmt], type_ignores=[])
        before = len(H._results)
        try:
            with cap():
                exec(compile(mod, "<checks>", "exec"), ns)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            info = error_info(e, "checks", focus=("<solution>",))
            msg = f"{info['type']}: {info['message']}"
            todo_caused = "TODO placeholder" in msg or "_Todo" in msg
            where = f"raised from your code, line {info['line']}" if info["line"] else None
            if todo_caused:
                funcs = _user_funcs_called(stmt, ns)
                head = (f"not filled in yet: {_ticks(funcs)} still {'has' if len(funcs) == 1 else 'have'} a TODO"
                        if funcs else "not filled in yet: a TODO is still in your code")
                details = [head + (f" (line {info['line']})" if info["line"] else "")]
            else:
                details = [msg]
                if user_has_todo:
                    details.append("(your code still has TODO placeholders, which may be why)")
                if where:
                    details.append(where)

            if is_check:
                label = stmt_label(stmt, src, ns)
                if len(H._results) > before:  # a loop of checks that stopped part-way
                    label += " (stopped part-way)"
                H._record(label, False, details)
                continue

            # A setup step failed.
            seg = (ast.get_source_segment(src, stmt) or "").strip().split("\n")[0]
            if todo_caused and stored:
                # Unfinished code, not a bug: let the dependent checks say so.
                for n in stored:
                    unavailable[n] = details[0]
                continue
            # A real error. Show it once, then point the dependent checks at it.
            H._record(f"setting up the checks: {seg[:70]}", False, details)
            for n in stored:
                unavailable[n] = f"skipped: needs `{n}`, which the failing set-up step above could not compute"
            continue

        for n in stored:
            unavailable.pop(n, None)
        # "still TODO" from inside check(): the value came back from a function
        # the learner wrote. Name it.
        new = H._results[before:]
        if any(r["details"] == ["not filled in yet: still TODO"] for r in new):
            funcs = _user_funcs_called(stmt, ns)
            msg = None
            if funcs:
                msg = f"not filled in yet: {_ticks(funcs)} still {'returns' if len(funcs) == 1 else 'return'} TODO"
            else:
                # Name the checked expression itself, e.g. `m["max_abs"]`.
                calls = _check_calls(stmt)
                if len(calls) == 1 and len(calls[0].args) >= 2:
                    seg = ast.get_source_segment(src, calls[0].args[1]) or ""
                    if seg and "\n" not in seg and len(seg) <= 60:
                        msg = f"not filled in yet: `{seg}` is still TODO"
            if msg:
                for r in new:
                    if r["details"] == ["not filled in yet: still TODO"]:
                        r["details"] = [msg]


# --------------------------------------------------------------------------- #
# Predict items
# --------------------------------------------------------------------------- #

def parse_shape(s):
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace("torch.Size", "").replace("×", "x")
    if not re.fullmatch(r"[\d\s,x()\[\]]*", s):
        raise ValueError("write a shape like (2, 3), or () for a scalar")
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", ",", s)
    inner = s.strip()
    if inner[:1] in "([" and inner[-1:] in ")]":
        inner = inner[1:-1].strip()
    if not inner:
        return ()
    if re.search(r"[()\[\]]", inner):
        raise ValueError(f"{s.strip()!r} is not one shape; write a single tuple like (2, 3)")
    parts = [p for p in re.split(r"[,\s]+", inner) if p]
    return tuple(int(p) for p in parts)


def run_predict(job, ns, cap, H, torch):
    answers = job.get("answers") or {}
    for item in job.get("items") or []:
        key, expr, kind = item["key"], item["expr"], item.get("kind", "shape")
        label = f"{key}  {expr}"
        try:
            with cap():
                val = eval(compile(expr, f"<item {key}>", "eval"), ns)
            truth_raise = "ok"
        except Exception as e:  # noqa: BLE001
            val, truth_raise = e, "raise"
        given = answers.get(key, "")
        if kind == "raises":
            g = (given or "").strip().lower()
            if g not in ("ok", "raise"):
                H._record(label, False, ["not answered yet"])
            elif g != truth_raise:
                extra = f" ({type(val).__name__}: {str(val).split(chr(10))[0][:140]})" if truth_raise == "raise" else ""
                H._record(label, False, [f"you said {g!r}, it is {truth_raise!r}{extra}"])
            else:
                extra = f": {type(val).__name__}: {str(val).split(chr(10))[0][:140]}" if truth_raise == "raise" else ""
                H._record(label, True, [f"-> {truth_raise}{extra}"])
            continue
        # shape
        if truth_raise == "raise":
            H._record(label, False, [f"authoring error: expression raised {type(val).__name__}: {val}"])
            continue
        truth = tuple(val.shape) if hasattr(val, "shape") else None
        try:
            mine = parse_shape(given)
        except ValueError as e:
            H._record(label, False, [f"could not read {given!r}: {e}"])
            continue
        if mine is None:
            H._record(label, False, ["not answered yet"])
        elif mine != truth:
            H._record(label, False, [f"you said {mine}, it is {truth}"])
        else:
            H._record(label, True, [f"= {truth}"])


# --------------------------------------------------------------------------- #

def run_job(job):
    import torch
    import torch.nn.functional as F
    import kata_helpers as H

    t_start = time.perf_counter()
    mode = job.get("mode", "run")
    device = job.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return {"status": "error", "error": {"phase": "setup", "type": "DeviceError",
                "message": "CUDA was requested but torch.cuda.is_available() is False",
                "traceback": "", "line": None}, "checks": [], "console": []}

    H._reset(device)
    if device != "cpu":
        torch.set_default_device(device)
    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=110)

    ns = {"__name__": "__main__", "__builtins__": builtins, "torch": torch, "F": F,
          "math": math, "DEVICE": device}
    for k in H.__all__:
        ns[k] = getattr(H, k)
    sys.stdin = io.StringIO("")

    out = {"status": "failed", "error": None, "checks": [], "console": [], "inspect": [],
           "uses": [], "device": device, "mode": mode}

    def finish(status=None):
        if status:
            out["status"] = status
        out["checks"] = list(H._results)
        out["timing"]["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return out

    out["timing"] = {}
    setup_cap = Cap()

    # Module-wide setup shared by every problem in the module.
    err = run_src(job.get("shared", ""), ns, "<shared>", setup_cap, "setup", focus=())
    if err:
        out["error"] = err
        return finish("error")

    if mode == "demo":
        for i, d in enumerate(job.get("prior_demos") or []):
            run_src(d, ns, f"<demo {i}>", Cap(), "demo", focus=())
        cap = Cap()
        err = run_echo(job.get("demo", ""), ns, "<demo>", cap, "demo")
        out["output"] = cap.text()
        out["error"] = err
        return finish("error" if err else "passed")

    # Earlier problems this one builds on (capstone): yours if it runs, else reference.
    for u in job.get("uses") or []:
        rep = {"id": u["id"], "title": u.get("title", u["id"]), "source": u.get("source", "reference")}
        if u.get("source") == "yours":
            snap = dict(ns)
            e = run_src(u["code"], ns, f"<{u['id']}>", Cap(), "uses", focus=())
            if e:
                ns.clear()
                ns.update(snap)
                rep["source"] = "reference"
                rep["note"] = f"your saved code raised {e['type']}, so the reference was used"
                run_src(u.get("reference", ""), ns, f"<{u['id']}>", Cap(), "uses", focus=())
        else:
            e = run_src(u.get("reference") or u.get("code", ""), ns, f"<{u['id']}>", Cap(), "uses", focus=())
            if e:
                rep["note"] = f"reference raised {e['type']}: {e['message']}"
        out["uses"].append(rep)

    err = run_src(job.get("given", ""), ns, "<given>", setup_cap, "given", focus=())
    if setup_cap.text().strip():
        out["console"].append({"phase": "given", "text": setup_cap.text()})
    if err:
        out["error"] = err
        return finish("error")

    if mode == "predict":
        cap = Cap()
        run_predict(job, ns, cap, H, torch)
        res = H._results
        out["status"] = "passed" if res and all(r["passed"] for r in res) else "failed"
        return finish()

    # The learner's code.
    user = job.get("user", "")
    out["todo_lines"] = todo_lines(user)
    ucap = Cap()
    t0 = time.perf_counter()
    err = run_src(user, ns, "<solution>", ucap, "user")
    if device == "cuda":
        torch.cuda.synchronize()
    out["timing"]["user_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if mode == "scratch":
        if err:
            out["error"] = err
            out["error"]["phase"] = "user"
            return finish("error")
        cap = Cap()
        err = run_echo(job.get("scratch", ""), ns, "<scratch>", cap, "scratch")
        out["output"] = cap.text()
        out["error"] = err
        return finish("error" if err else "passed")

    if ucap.text():
        out["console"].append({"phase": "user", "text": ucap.text()})
    if err:
        out["error"] = err
        return finish("error")

    names = job.get("inspect") or starter_names(job.get("starter", ""))
    for n in names[:24]:
        if n in ns:
            try:
                out["inspect"].append(describe(n, ns[n], torch, H))
            except Exception as e:  # noqa: BLE001
                out["inspect"].append({"name": n, "kind": "error", "summary": str(e)})
        else:
            out["inspect"].append({"name": n, "kind": "missing", "summary": "not defined by your code"})

    ccap = Cap()
    t0 = time.perf_counter()
    run_checks(job.get("checks", ""), ns, ccap, H, bool(out["todo_lines"]))
    out["timing"]["checks_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if ccap.text().strip():
        out["console"].append({"phase": "checks", "text": ccap.text()})

    res = H._results
    out["status"] = "passed" if res and all(r["passed"] for r in res) else "failed"
    return finish()


def main():
    t0 = time.perf_counter()
    import torch
    import torch.nn.functional  # noqa: F401
    import kata_helpers  # noqa: F401
    if torch.cuda.is_available() and os.environ.get("KATA_WARM_CUDA", "1") == "1":
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    emit(READY, {"env": env_info(torch), "warm_ms": round((time.perf_counter() - t0) * 1000)})

    line = sys.stdin.readline()
    if not line.strip():
        return
    job = json.loads(line)
    try:
        result = run_job(job)
    except BaseException as e:  # noqa: BLE001
        result = {"status": "error", "checks": [], "console": [],
                  "error": {"phase": "harness", "type": type(e).__name__, "message": str(e),
                            "traceback": traceback.format_exc(), "line": None}}
    sys.stdout, sys.stderr = REAL_STDOUT, sys.__stderr__
    emit(RESULT, result)


if __name__ == "__main__":
    main()

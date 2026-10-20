"""Static policy for algorithm code and the loader used by the trainer.

An algorithm directory may contain several ``.py`` files; ``algorithm.py`` must define ``ALGORITHM``.
Every file is parsed and checked before anything is imported:

* imports only from ``ALLOWED_MODULES`` (and sibling modules of the same directory),
* no calls to dangerous builtins (``open``, ``exec``, ``eval``, ``compile``, ``__import__``, ``getattr`` ...),
* no dunder attribute access other than ``__init__`` / ``__name__`` style definitions on ``self``,
* no attribute names that reach files, processes, the network, the autograd graph or interpreter internals.

This is a guard against accidental or casual leakage, not a security boundary; official scoring additionally
runs training without access to evaluation data and evaluates in a separate process that never imports the
algorithm (see ``selectrl.bench.score``).

    python -m selectrl.bench.guard algorithms/candidate      # exit code 0 = OK
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ALLOWED_MODULES = {
    "numpy", "math", "cmath", "torch", "torch.nn", "torch.nn.functional",
    "itertools", "functools", "collections", "dataclasses", "typing", "fractions", "decimal", "heapq", "bisect",
    "statistics", "operator", "enum", "abc", "__future__", "numbers", "copy",
    "selectrl.algo_api", "selectrl.algolib", "selectrl.algolib.combin", "selectrl.algolib.selector",
    "selectrl.algolib.sspo_dp",
}
BANNED_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "exit", "quit", "help", "memoryview", "object.__getattribute__",
}
BANNED_ATTRS = {
    # files / processes / network / serialization
    "load", "save", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "memmap", "fromregex", "DataSource",
    "lib", "ctypeslib", "ctypes", "npyio", "from_file", "hub", "ops", "library", "jit", "package", "distributed",
    "multiprocessing", "utils", "cuda", "backends", "_C", "classes", "testing", "compiler", "export", "onnx",
    "storage", "untyped_storage", "share_memory_", "set_", "data_ptr",
    # autograd graph and hooks (could reach the optimizer's parameters)
    "grad_fn", "next_functions", "variable", "register_hook", "register_forward_hook", "register_backward_hook",
    "register_full_backward_hook", "register_forward_pre_hook", "backward", "autograd", "retain_grad",
    "requires_grad_", "grad", "_base", "_version", "data",
    # interpreter internals and module re-exports that lead back to os / sys / importlib
    "f_globals", "f_locals", "f_back", "gi_frame", "co_code", "func_globals", "mro",
    "sys", "os", "modules", "builtins", "importlib", "subprocess", "system", "popen", "spawn", "exec_module",
    "loader", "f2py", "distutils", "compile", "serialization", "_dynamo", "fx", "inspect", "gc", "core", "_core",
    "attrgetter", "methodcaller", "profiler", "overrides", "socket", "shutil", "pathlib", "io", "tempfile",
}
ALLOWED_DUNDERS = {"__init__", "__name__", "__doc__", "__post_init__", "__call__", "__repr__", "__len__",
                   "__future__"}


class GuardError(Exception):
    pass


def _check_file(path: Path, siblings: set) -> List[str]:
    errs: List[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        return [f"{path.name}:{e.lineno}: syntax error: {e.msg}"]

    def bad(node: ast.AST, msg: str) -> None:
        errs.append(f"{path.name}:{getattr(node, 'lineno', '?')}: {msg}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_MODULES and a.name.split(".")[0] not in siblings:
                    bad(node, f"import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:  # relative import of a sibling file
                if mod and mod.split(".")[0] not in siblings:
                    bad(node, f"relative import '{mod}' is not a file of this algorithm")
                continue
            if mod not in ALLOWED_MODULES and mod.split(".")[0] not in siblings:
                bad(node, f"import from '{mod}' is not allowed")
            for a in node.names:
                if a.name == "*":
                    bad(node, "star imports are not allowed")
                elif a.name in BANNED_ATTRS or a.name in BANNED_CALLS:
                    bad(node, f"importing '{a.name}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_CALLS or (node.id.startswith("__") and node.id.endswith("__")
                                           and node.id not in ALLOWED_DUNDERS):
                bad(node, f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("__") and a.endswith("__") and a not in ALLOWED_DUNDERS:
                bad(node, f"dunder attribute '.{a}' is not allowed")
            elif a in BANNED_ATTRS:
                bad(node, f"attribute '.{a}' is not allowed in algorithm code")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bad(node, "global/nonlocal statements are not allowed")
    return errs


def check_algorithm_dir(algo_dir: Path) -> Tuple[List[str], str]:
    """Returns (errors, sha256 over all .py files in the directory)."""
    algo_dir = Path(algo_dir)
    files = sorted(p for p in algo_dir.rglob("*.py") if "__pycache__" not in p.parts)
    if not (algo_dir / "algorithm.py").exists():
        return [f"{algo_dir}/algorithm.py not found"], ""
    others = [p for p in algo_dir.rglob("*") if p.is_file() and p.suffix not in (".py", ".md", ".pyc")
              and "__pycache__" not in p.parts]
    errs = [f"non-Python file not allowed in the algorithm directory: {p.relative_to(algo_dir)}" for p in others]
    siblings = {p.stem for p in files}
    h = hashlib.sha256()
    for p in files:
        errs += _check_file(p, siblings)
        h.update(str(p.relative_to(algo_dir)).encode() + b"\0" + p.read_bytes() + b"\0")
    return errs, h.hexdigest()


def load_algorithm(algo_dir: Path):
    """Guard-check then import ``algorithm.py`` and return ``(ALGORITHM instance, info dict)``."""
    from selectrl.algo_api import LOSS_AGG_MODES, Algorithm

    algo_dir = Path(algo_dir).resolve()
    errs, digest = check_algorithm_dir(algo_dir)
    if errs:
        raise GuardError("algorithm rejected by the guard:\n  " + "\n  ".join(errs))
    pkg = "selectrl_algo_" + digest[:12]
    spec = importlib.util.spec_from_file_location(pkg, algo_dir / "algorithm.py",
                                                  submodule_search_locations=[str(algo_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    algo = getattr(mod, "ALGORITHM", None)
    if not isinstance(algo, Algorithm):
        raise GuardError("algorithm.py must define ALGORITHM = <instance of selectrl.algo_api.Algorithm>")
    if algo.loss_agg_mode not in LOSS_AGG_MODES:
        raise GuardError(f"loss_agg_mode must be one of {LOSS_AGG_MODES}")
    if algo.kl_coef > 0 and not algo.needs_ref_logp:
        raise GuardError("kl_coef > 0 requires needs_ref_logp = True")
    return algo, {"sha256": digest, "name": algo.name, "dir": str(algo_dir)}


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m selectrl.bench.guard <algorithm_dir>")
        sys.exit(2)
    errs, digest = check_algorithm_dir(Path(sys.argv[1]))
    if errs:
        print("REJECTED:\n  " + "\n  ".join(errs))
        sys.exit(1)
    print(f"OK sha256={digest}")


if __name__ == "__main__":
    main()

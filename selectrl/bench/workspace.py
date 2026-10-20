"""Agent workspaces and submissions.

    python -m selectrl.bench.workspace make --out ../selectrl_agent_ws
        Copy of the repository for a benchmark agent: hidden test problems, their checker cases and every cached
        execution record of test problems are removed; every file except algorithms/ is made read-only.
    python -m selectrl.bench.workspace collect --workspace ../selectrl_agent_ws --name my_algo
        Copy the agent's algorithms/candidate into submissions/<name>/ of this (pristine) repository after the
        guard accepts it. Score it with: bash scripts/score.sh submissions/<name>
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
from pathlib import Path

from selectrl.bench.guard import check_algorithm_dir
from selectrl.config import REPO_ROOT

EXCLUDE_TOP = {".git", "runs", "submissions", "models", "logs", "__pycache__", ".pytest_cache"}


def _test_pids() -> set:
    p = REPO_ROOT / "data" / "splits" / "test.txt"
    return {l.strip() for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


def make(out: Path, link_models: bool = True) -> None:
    out = out.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    hidden = _test_pids()

    def ignore(dirpath: str, names):
        rel = Path(dirpath).resolve().relative_to(REPO_ROOT)
        skip = set()
        for n in names:
            if rel == Path(".") and n in EXCLUDE_TOP:
                skip.add(n)
            elif n in ("__pycache__",):
                skip.add(n)
            elif rel == Path("data/problems") and n.startswith("test."):
                skip.add(n)
            elif rel == Path("data/splits") and n.startswith("test."):
                skip.add(n)
            elif rel.parts[:1] == ("cache",) and n.split(".")[0] in hidden:
                skip.add(n)
            elif rel.parts[:1] == ("results",) and n.startswith("test_") and n.endswith((".jsonl", ".jsonl.gz")):
                skip.add(n)
        return skip

    shutil.copytree(REPO_ROOT, out, ignore=ignore, symlinks=True)
    if link_models and (REPO_ROOT / "models").exists():
        os.symlink(REPO_ROOT / "models", out / "models")
    (out / "runs").mkdir()
    for root, dirs, files in os.walk(out):
        rp = Path(root)
        if rp == out / "runs" or (out / "algorithms") in rp.parents or rp == out / "algorithms" \
                or (out / "runs") in rp.parents:
            continue
        for f in files:
            fp = rp / f
            if fp.is_symlink():
                continue
            fp.chmod(fp.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"agent workspace ready: {out}  (hidden test problems removed: {len(hidden)})")


def collect(workspace: Path, name: str) -> Path:
    src = workspace.resolve() / "algorithms" / "candidate"
    errs, digest = check_algorithm_dir(src)
    if errs:
        raise SystemExit("submission rejected by the guard:\n  " + "\n  ".join(errs))
    dst = REPO_ROOT / "submissions" / name
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dst / "SUBMISSION.md").write_text(f"name: {name}\nsha256 of the collected .py files: {digest}\n")
    print(f"collected {src} -> {dst} (sha256 {digest[:16]})")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--no-model-link", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--workspace", required=True)
    c.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        make(Path(a.out), link_models=not a.no_model_link)
    else:
        collect(Path(a.workspace), a.name)


if __name__ == "__main__":
    main()

"""sha256 lock of everything a benchmark agent must not change.

    python -m selectrl.bench.integrity write     # maintainers: regenerate bench/LOCK.json
    python -m selectrl.bench.integrity verify    # exit 0 iff every locked file matches (test data may be absent)

Locked: the harness package, reference algorithms, protocol configs, scripts, split lists and problem files, and
the checker cache (hashed through its per-file manifest). The agent-editable directory ``algorithms/`` and the
outputs (``runs/``, ``results/``) are not locked.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

from selectrl.config import REPO_ROOT

LOCK_PATH = REPO_ROOT / "bench" / "LOCK.json"
LOCKED_GLOBS = [
    "selectrl/**/*.py", "reference_algorithms/**/*.py", "configs/protocol/*.yaml", "scripts/*.sh",
    "data/splits/*.txt", "data/problems/train.jsonl.gz", "data/problems/val.jsonl.gz", "data/problems/test.jsonl.gz",
    "data/filter_stats.json", "data/active/base_pass_summary.json", "cache/checkers/MANIFEST.json",
    "pyproject.toml",
]
# files that are legitimately absent in an agent workspace (hidden test data)
OPTIONAL = {"data/problems/test.jsonl.gz", "data/splits/test.txt"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_hashes() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pat in LOCKED_GLOBS:
        for p in sorted(REPO_ROOT.glob(pat)):
            if p.is_file() and "__pycache__" not in p.parts:
                out[str(p.relative_to(REPO_ROOT))] = sha256_file(p)
    return out


def write_lock() -> Dict[str, str]:
    hashes = current_hashes()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps({"files": hashes}, indent=1, sort_keys=True) + "\n")
    return hashes


def verify(allow_missing_optional: bool = True) -> List[str]:
    if not LOCK_PATH.exists():
        return [f"{LOCK_PATH.relative_to(REPO_ROOT)} missing"]
    locked = json.loads(LOCK_PATH.read_text())["files"]
    cur = current_hashes()
    problems = []
    for rel, h in locked.items():
        if rel not in cur:
            if allow_missing_optional and rel in OPTIONAL:
                continue
            problems.append(f"missing locked file: {rel}")
        elif cur[rel] != h:
            problems.append(f"modified locked file: {rel}")
    for rel in cur:
        if rel not in locked:
            problems.append(f"unexpected new file in a locked location: {rel}")
    return problems


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "write":
        n = len(write_lock())
        print(f"wrote {LOCK_PATH.relative_to(REPO_ROOT)} ({n} files)")
        return
    problems = verify()
    if problems:
        print("INTEGRITY FAILED:\n  " + "\n  ".join(problems[:50]) + ("\n  ..." if len(problems) > 50 else ""))
        sys.exit(1)
    print("integrity OK")


if __name__ == "__main__":
    main()

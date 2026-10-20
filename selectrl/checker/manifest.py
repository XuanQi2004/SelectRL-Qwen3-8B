"""Manifest of the checker cache (one sha256 per artifacts / cases file) so the lock can cover it cheaply.

    python -m selectrl.checker.manifest write|verify
"""
from __future__ import annotations

import json
import sys

from selectrl.config import REPO_ROOT
from selectrl.utils import file_sha256

ROOT = REPO_ROOT / "cache" / "checkers"
MANIFEST = ROOT / "MANIFEST.json"


def current() -> dict:
    out = {}
    for sub in ("artifacts", "cases"):
        for p in sorted((ROOT / sub).glob("*.json.gz")):
            out[f"{sub}/{p.name}"] = file_sha256(p)
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    cur = current()
    if cmd == "write":
        MANIFEST.write_text(json.dumps(cur, indent=0, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST} ({len(cur)} files)")
        return
    if not MANIFEST.exists():
        print("no manifest"); sys.exit(1)
    want = json.loads(MANIFEST.read_text())
    bad = [k for k, v in want.items() if cur.get(k) != v]
    missing = [k for k in want if k not in cur]
    print(f"{len(want)} files in manifest, {len(missing)} missing, {len(bad) - len(missing)} modified")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""Build the execution-check cases of every problem from its generated helper programs.

Ported from SSPO v1 (ec_build.py).

Per problem:
1. Validators are kept only if they print VALID on every public example input.
2. Each generator runs with MODE SEED = max 1-2, small 1-6, edge 1-2; empty outputs and outputs larger than 8 MiB
   are dropped, the rest de-duplicated by content (first occurrence wins).
3. An input is valid if every kept validator prints VALID on it (``None`` = unverified when none was kept).
4. Brute-force solutions are kept only if they pass every public example; the reference of a valid or unverified
   small / edge input is the plurality output key of the kept brute-force solutions that finish
   (``ref_agree >= 2`` = confirmed reference used by Chk).
Output: ``<cache>/checkers/cases/<pid>.json.gz`` (resumable; rebuilt when its ``checker_version`` differs).

Usage: python -m selectrl.checker.build --problems data/problems/train_candidates.jsonl.gz [--cache cache]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import re
from pathlib import Path
from typing import Any, Iterable

from selectrl.sandbox import out_key, pmap, run_prog
from selectrl.utils import atomic_write_json_gz, iter_jsonl, log, read_json_gz, sha1_16

CHECKER_VERSION = "ccv1"
RUNS: tuple[tuple[str, int], ...] = (
    (("max", 1), ("max", 2)) + tuple(("small", s) for s in range(1, 7)) + (("edge", 1), ("edge", 2))
)
MAX_INPUT_BYTES = 8 << 20
EXAMPLE_TIMEOUT_S = 10.0
GEN_TIMEOUT_S = 30.0
VAL_TIMEOUT_S = 30.0
BF_TIMEOUT_S = 10.0
VAL_OUT_LIMIT = 1 << 16


def cases_path(cases_dir: str | Path, pid: str) -> Path:
    """Path of a problem's cases file."""
    return Path(cases_dir) / f"{pid}.json.gz"


def has_current_cases(cases_dir: str | Path, pid: str) -> bool:
    """True if the problem's cases file exists and was built by this checker version (reads only its header)."""
    p = cases_path(cases_dir, pid)
    if not p.exists():
        return False
    with gzip.open(p, "rt", encoding="utf-8") as f:
        m = re.search(r'"checker_version": "([^"]*)"', f.read(512))
    return bool(m) and m.group(1) == CHECKER_VERSION


def load_cases(cases_dir: str | Path, pid: str) -> dict | None:
    """The problem's cases of the current checker version, or ``None`` if missing or stale."""
    p = cases_path(cases_dir, pid)
    if not p.exists():
        return None
    cases = read_json_gz(p)
    return cases if cases.get("checker_version") == CHECKER_VERSION else None


def _is_valid_verdict(r: tuple[str, str, float, float]) -> bool:
    return r[0] == "ok" and r[1].strip() == "VALID"


def build_cases(problem: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    """Build the execution-check cases and confirmed references of one problem."""
    ex_in = [t[0] for t in problem["public"]]
    ex_out = [t[1] for t in problem["public"]]

    vals, val_info = [], []
    for code in artifacts.get("val", []):
        if not code:
            val_info.append("nocode")
            continue
        ok = all(_is_valid_verdict(r) for r in run_prog(code, ex_in, EXAMPLE_TIMEOUT_S, keep_out=True))
        val_info.append("kept" if ok else "rejected_examples")
        if ok:
            vals.append(code)

    raw, gen_info = [], []
    for gi, code in enumerate(artifacts.get("gen", [])):
        if not code:
            gen_info.append("nocode")
            continue
        res = run_prog(code, [f"{m} {s}\n" for m, s in RUNS], GEN_TIMEOUT_S, keep_out=True, out_limit=MAX_INPUT_BYTES)
        n_ok = 0
        for (mode, seed), r in zip(RUNS, res):
            if r[0] == "ok" and r[1].strip():
                inp = r[1] if r[1].endswith("\n") else r[1] + "\n"
                if len(inp.encode("utf-8")) <= MAX_INPUT_BYTES:
                    raw.append({"mode": mode, "gen": gi, "seed": seed, "inp": inp})
                    n_ok += 1
        gen_info.append(f"{n_ok}/{len(RUNS)}")

    seen: set[str] = set()
    inputs: list[dict[str, Any]] = []
    for x in raw:
        h = sha1_16(x["inp"])
        if h not in seen:
            seen.add(h)
            inputs.append({"id": h, **x})

    verdicts = [run_prog(v, [x["inp"] for x in inputs], VAL_TIMEOUT_S, keep_out=True, out_limit=VAL_OUT_LIMIT)
                for v in vals]
    for k, x in enumerate(inputs):
        x["valid"] = all(_is_valid_verdict(vr[k]) for vr in verdicts) if vals else None

    bfs, bf_info = [], []
    for code in artifacts.get("bf", []):
        if not code:
            bf_info.append("nocode")
            continue
        ok = all(r[0] == "pass" for r in run_prog(code, ex_in, EXAMPLE_TIMEOUT_S, expected=ex_out))
        bf_info.append("kept" if ok else "failed_examples")
        if ok:
            bfs.append(code)

    small = [x for x in inputs if x["mode"] != "max" and x["valid"] is not False]
    outs = [run_prog(b, [x["inp"] for x in small], BF_TIMEOUT_S, keep_out=True) for b in bfs]
    for x in inputs:
        x.update(ref=None, ref_key=None, ref_agree=0, n_bf_ok=0)
    for k, x in enumerate(small):
        oks = [o[k][1] for o in outs if o[k][0] == "ok"]
        x["n_bf_ok"] = len(oks)
        if oks:
            keys = [out_key(o) for o in oks]
            key, n = collections.Counter(keys).most_common(1)[0]
            x.update(ref=oks[keys.index(key)], ref_key=key, ref_agree=n)

    for x in inputs:
        if x["valid"] is False and x["mode"] == "max":
            x["inp"] = None  # keep only the metadata of rejected large inputs

    return {
        "pid": problem["pid"],
        "checker_version": CHECKER_VERSION,
        "val_info": val_info,
        "gen_info": gen_info,
        "bf_info": bf_info,
        "n_val_kept": len(vals),
        "n_bf_kept": len(bfs),
        "bf_codes": bfs,
        "inputs": inputs,
    }


def case_stats(cases: dict[str, Any]) -> dict[str, int]:
    """Counts of usable stress inputs, usable small inputs and confirmed references."""
    usable = [x for x in cases["inputs"] if x["valid"] is not False and x["inp"]]
    return {
        "n_max": sum(x["mode"] == "max" for x in usable),
        "n_small": sum(x["mode"] != "max" for x in usable),
        "n_confirmed": sum(x["mode"] != "max" and x["ref_agree"] >= 2 for x in usable),
    }


def build_all(problems: Iterable[dict[str, Any]], artifacts_dir: str | Path, cases_dir: str | Path) -> dict[str, Any]:
    """Build (or reuse) the cases of every problem whose artifacts exist; problems run in parallel on the pool."""
    probs = list(problems)
    todo = [p for p in probs if (Path(artifacts_dir) / f"{p['pid']}.json.gz").exists()
            and not has_current_cases(cases_dir, p["pid"])]
    n_missing = sum(not (Path(artifacts_dir) / f"{p['pid']}.json.gz").exists() for p in probs)
    log(f"[build] problems {len(probs)}, to build {len(todo)}, without artifacts {n_missing}")

    def go(problem: dict[str, Any]) -> tuple[str, dict[str, int] | None, str]:
        pid = problem["pid"]
        try:
            cases = build_cases(problem, read_json_gz(Path(artifacts_dir) / f"{pid}.json.gz"))
        except Exception as e:  # one broken problem must not stop the others; it is retried on the next run
            return pid, None, repr(e)[:300]
        atomic_write_json_gz(cases_path(cases_dir, pid), cases)
        return pid, case_stats(cases), ""

    built, failed = 0, {}
    for pid, stats, err in pmap(go, todo):
        if stats is None:
            failed[pid] = err
            log(f"[build] {pid} FAILED {err}")
        else:
            built += 1
    return {"n_problems": len(probs), "built": built, "reused": len(probs) - n_missing - len(todo),
            "missing_artifacts": n_missing, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build execution-check cases from generated helper programs.")
    ap.add_argument("--problems", required=True, help="problems .jsonl.gz")
    ap.add_argument("--cache", default="cache", help="cache root (reads checkers/artifacts, writes checkers/cases)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    probs = sorted(iter_jsonl(args.problems), key=lambda p: p["pid"])[args.shard :: args.nshards]
    root = Path(args.cache) / "checkers"
    log(f"[build] {build_all(probs, root / 'artifacts', root / 'cases')}")


if __name__ == "__main__":
    main()

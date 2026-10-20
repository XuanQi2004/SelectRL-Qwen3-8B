"""Tables and paired comparisons over finished runs.

    python -m selectrl.eval.report --out results/leaderboard.md [--runs runs/* results/baselines/*]

A run directory is any directory with ``eval/<split>_final.summary.json`` (plus the per-problem ``.jsonl``).
Paired differences use a contest-clustered bootstrap over the problems both runs evaluated (2000 resamples,
fixed seed), which respects the correlation of problems from the same contest.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from selectrl.config import resolve


def load_rows(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return {r["pid"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def metric_of(row: dict, metric: str) -> Optional[float]:
    kind, k = metric.split("@")
    return row.get(kind, {}).get(k)


def paired_bootstrap(a: Dict[str, dict], b: Dict[str, dict], metric: str, n_boot: int = 2000,
                     seed: int = 0) -> Optional[Tuple[float, float, float, int]]:
    """Mean of (a - b) over common problems with a contest-clustered 95% bootstrap interval."""
    common = sorted(set(a) & set(b))
    pairs = [(a[p].get("contest_id") or p, metric_of(a[p], metric), metric_of(b[p], metric)) for p in common]
    pairs = [x for x in pairs if x[1] is not None and x[2] is not None]
    if not pairs:
        return None
    by_c: Dict[str, List[float]] = {}
    for c, x, y in pairs:
        by_c.setdefault(c, []).append(x - y)
    clusters = list(by_c.values())
    sums = np.array([sum(v) for v in clusters])
    cnts = np.array([len(v) for v in clusters])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    boots = sums[idx].sum(1) / np.maximum(1, cnts[idx].sum(1))
    mean = float(sums.sum() / cnts.sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi), len(pairs)


def find_runs(patterns: List[str]) -> List[Path]:
    out = []
    for pat in patterns:
        for d in sorted(glob.glob(str(resolve(pat)))):
            p = Path(d)
            if (p / "eval").is_dir():
                out.append(p)
    return out


def build(patterns: List[str], metric: str, reference: Optional[str]) -> str:
    runs = find_runs(patterns)
    lines = [f"| run | algorithm | seed | val {metric} | test {metric} | test pass@8 | test pass@1 | "
             f"test chk rate | Δ test vs `{reference}` [95% CI] |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    ref_rows = {}
    if reference:
        ref_dir = next((r for r in runs if r.name == reference), None)
        if ref_dir:
            ref_rows = load_rows(ref_dir / "eval" / "test_final.jsonl")
    for r in runs:
        man = json.loads((r / "manifest.json").read_text()) if (r / "manifest.json").exists() else {}
        sv = r / "eval" / "val_final.summary.json"
        st = r / "eval" / "test_final.summary.json"
        v = json.loads(sv.read_text()) if sv.exists() else {}
        t = json.loads(st.read_text()) if st.exists() else {}
        fmt = lambda x: f"{100 * x:.2f}" if isinstance(x, (int, float)) else "–"  # noqa: E731
        delta = "–"
        if ref_rows and r.name != reference:
            res = paired_bootstrap(load_rows(r / "eval" / "test_final.jsonl"), ref_rows, metric)
            if res:
                delta = f"{100 * res[0]:+.2f} [{100 * res[1]:+.2f}, {100 * res[2]:+.2f}] (n={res[3]})"
        lines.append(f"| `{r.name}` | {man.get('algo_name', 'base')} | {man.get('seed', '–')} | {fmt(v.get(metric))} | "
                     f"{fmt(t.get(metric))} | {fmt(t.get('pass@8'))} | {fmt(t.get('pass@1'))} | "
                     f"{fmt(t.get('chk_rate'))} | {delta} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=["runs/*", "results/baselines/*"])
    ap.add_argument("--metric", default="select@8")
    ap.add_argument("--reference", default="frozen_base")
    ap.add_argument("--out", default="results/leaderboard.md")
    a = ap.parse_args()
    md = "# Leaderboard (numbers in %)\n\n" + build(a.runs, a.metric, a.reference)
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()

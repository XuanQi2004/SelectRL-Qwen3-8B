"""Official scoring of a finished run (called by scripts/score.sh after training and evaluation).

    python -m selectrl.bench.score --run-dir runs/<name>

Checks: the integrity lock, the guard on the algorithm files recorded in the manifest, that the run used the
official protocol (hash of configs/protocol/qwen3_8b_40960.yaml), that it finished every step, and that the test and
validation evaluations of the final checkpoint exist with the protocol's n_eval. Writes ``<run>/score.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from selectrl.bench.guard import check_algorithm_dir
from selectrl.bench.integrity import verify
from selectrl.config import load_protocol, protocol_hash, resolve
from selectrl.eval.report import load_rows, paired_bootstrap
from selectrl.utils import atomic_write_json

OFFICIAL = "configs/protocol/qwen3_8b_40960.yaml"


def score(run_dir: Path, protocol: str = OFFICIAL, baselines: str = "results/baselines") -> dict:
    cfg = load_protocol(protocol)
    problems = []
    integ = verify()
    if integ:
        problems += [f"integrity: {p}" for p in integ]
    man = json.loads((run_dir / "manifest.json").read_text())
    if man.get("protocol_hash") != protocol_hash(cfg):
        problems.append(f"run used protocol hash {man.get('protocol_hash')}, official is {protocol_hash(cfg)}")
    errs, digest = check_algorithm_dir(resolve(man["algo_dir"]))
    if errs:
        problems += [f"guard: {e}" for e in errs]
    if digest != man.get("algo_sha256"):
        problems.append("algorithm files changed after training")
    done = run_dir / "done"
    if not done.exists():
        problems.append("training did not finish (no done marker)")
    out = {"run": run_dir.name, "algo_name": man.get("algo_name"), "algo_sha256": man.get("algo_sha256"),
           "seed": man.get("seed"), "protocol": cfg["protocol"], "protocol_hash": protocol_hash(cfg)}
    metric = cfg["eval"]["primary_metric"]
    for split in ("test", "val"):
        sp = run_dir / "eval" / f"{split}_final.summary.json"
        if not sp.exists():
            problems.append(f"missing {sp.relative_to(run_dir)}")
            continue
        s = json.loads(sp.read_text())
        if s.get("n_eval") != cfg["eval"]["n_eval"] or s.get("protocol_hash") != protocol_hash(cfg):
            problems.append(f"{split} evaluation does not follow the protocol")
        out[split] = {k: s.get(k) for k in (metric, "pass@1", "pass@8", "select@1", "select@4", "select@16",
                                            "chk_rate", "n_problems")}
    rows = load_rows(run_dir / "eval" / "test_final.jsonl")
    out["comparisons"] = {}
    for b in sorted(resolve(baselines).glob("*")):
        brows = load_rows(b / "eval" / "test_final.jsonl")
        res = paired_bootstrap(rows, brows, metric) if rows and brows else None
        if res:
            out["comparisons"][b.name] = {"delta": res[0], "ci95": [res[1], res[2]], "n_problems": res[3]}
    out["valid"] = not problems
    out["problems"] = problems
    out["score"] = out.get("test", {}).get(metric) if not problems else None
    atomic_write_json(run_dir / "score.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--protocol", default=OFFICIAL)
    a = ap.parse_args()
    res = score(resolve(a.run_dir), a.protocol)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["valid"] else 1)


if __name__ == "__main__":
    main()

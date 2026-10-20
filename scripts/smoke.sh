#!/usr/bin/env bash
# 15-30 minute end-to-end check with the real model on this machine: 2 GRPO steps (4 problems x 4 rollouts),
# resume, and a tiny evaluation (8 validation problems x 4 samples). Prints PASS at the end.
#
#   bash scripts/smoke.sh [--gpus 0,1,...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPUS=""; [[ "${1:-}" == "--gpus" ]] && GPUS="$2"
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,5p' "$0"; exit 0; }
rm -rf runs/smoke_grpo
# the first attempt stops after step 1 on purpose; train.sh then restarts and must resume at step 2
SELECTRL_STOP_AFTER_STEP=1 bash scripts/train.sh reference_algorithms/grpo --name smoke_grpo \
  --protocol configs/protocol/smoke.yaml ${GPUS:+--gpus "${GPUS}"}
grep -q "resumed after step 1" runs/smoke_grpo/logs/train.log || { echo "SMOKE FAIL: resume path not exercised"; exit 1; }
${PYTHON} - <<'PY'
import json, pathlib, sys
run = pathlib.Path("runs/smoke_grpo")
rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
ev = json.loads((run / "eval" / "val_final.summary.json").read_text())
ok = len(rows) == 2 and (run / "done").exists() and ev["n_problems"] > 0
print(json.dumps({"steps": [{k: r[k] for k in ("step", "g_rate", "chk_rate", "loss", "mismatch_abs_lp", "t_step_s") if k in r} for r in rows],
                  "val": {k: ev.get(k) for k in ("n_problems", "pass@1", "select@2", "chk_rate")}}, indent=1))
print("SMOKE PASS" if ok else "SMOKE FAIL")
sys.exit(0 if ok else 1)
PY

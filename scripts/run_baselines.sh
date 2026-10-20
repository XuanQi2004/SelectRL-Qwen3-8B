#!/usr/bin/env bash
# Reproduce the reference results: frozen base model + GRPO, Pass@k and SSPO (seed 0 by default).
# Every stage skips work that is already finished, so re-running this command continues after an interruption.
#
#   bash scripts/run_baselines.sh [--seeds "0 1"] [--algos "grpo passk sspo"] [--gpus 0,1,...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEEDS="0"; ALGOS="sspo grpo passk"; GPUS=""
while [[ $# -gt 0 ]]; do
  case "$1" in --seeds) SEEDS="$2"; shift 2;; --algos) ALGOS="$2"; shift 2;; --gpus) GPUS="$2"; shift 2;;
    -h|--help) sed -n '2,6p' "$0"; exit 0;; *) echo "unknown argument $1" >&2; exit 2;; esac
done
bash scripts/eval.sh base --name frozen_base ${GPUS:+--gpus "${GPUS}"}
for S in ${SEEDS}; do
  for A in ${ALGOS}; do
    bash scripts/train.sh "reference_algorithms/${A}" --seed "${S}" --name "${A}_seed${S}" ${GPUS:+--gpus "${GPUS}"}
  done
done
${PYTHON} -m selectrl.eval.report --runs "runs/*" --reference frozen_base --out results/leaderboard_local.md
log "baselines finished; table: results/leaderboard_local.md; bundle for sharing: bash scripts/collect_results.sh"

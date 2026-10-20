#!/usr/bin/env bash
# Official scoring of an algorithm: integrity + guard checks, training with the locked protocol, evaluation of the
# final checkpoint on val and on the hidden test split, then runs/<name>/score.json.
#
#   bash scripts/score.sh <algorithm_dir> [--seed 0] [--name NAME] [--gpus 0,1,...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -ge 1 && "$1" != "-h" && "$1" != "--help" ]] || { sed -n '2,6p' "$0"; exit 0; }
ALGO="$1"; shift
SEED=0; NAME=""; GPUS=""
while [[ $# -gt 0 ]]; do
  case "$1" in --seed) SEED="$2"; shift 2;; --name) NAME="$2"; shift 2;; --gpus) GPUS="$2"; shift 2;;
    *) echo "unknown argument $1" >&2; exit 2;; esac
done
NAME="${NAME:-score_$(basename "${ALGO}")_seed${SEED}}"
${PYTHON} -m selectrl.bench.integrity verify
${PYTHON} -m selectrl.checker.manifest verify
[[ -f data/problems/test.jsonl.gz ]] || { echo "hidden test split missing: scoring must run in the pristine repository" >&2; exit 1; }
bash scripts/train.sh "${ALGO}" --seed "${SEED}" --name "${NAME}" ${GPUS:+--gpus "${GPUS}"}
${PYTHON} -m selectrl.bench.score --run-dir "runs/${NAME}"

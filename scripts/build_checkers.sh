#!/usr/bin/env bash
# Optional: (re)build the checker cache for the shipped splits. Not needed when the released cache is present.
# Helper programs are sampled from the base model through one vLLM server per GPU (inference only, so every GPU
# serves with gpu-memory-utilization 0.85 by default; set SELECTRL_EVAL_MEM_UTIL to change it), then the cases are
# built on CPU. Both stages skip problems that are already done.
#
#   bash scripts/build_checkers.sh [--gpus 0,1,...] [--splits "train val test"]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
GPUS=""; SPLITS="train val test"
while [[ $# -gt 0 ]]; do
  case "$1" in --gpus) GPUS="$2"; shift 2;; --splits) SPLITS="$2"; shift 2;;
    -h|--help) sed -n '2,8p' "$0"; exit 0;; *) echo "unknown argument $1" >&2; exit 2;; esac
done
FILES=""; for S in ${SPLITS}; do [[ -f "data/problems/${S}.jsonl.gz" ]] && FILES="${FILES} data/problems/${S}.jsonl.gz"; done
preset "${GPUS}"
ALL=$(${PYTHON} -c "print(','.join(sorted(set('${SELECTRL_SERVER_GPUS},${SELECTRL_TRAIN_GPUS}'.split(',')), key=int)))")
trap 'stop_servers' EXIT
start_servers runs/_checkers/servers.json runs/_checkers/logs "${ALL}" "${SELECTRL_EVAL_MEM_UTIL:-0.85}" "${SELECTRL_MAX_NUM_SEQS}"
${PYTHON} -m selectrl.checker.gen --servers runs/_checkers/servers.json --problems ${FILES}
stop_servers
for F in ${FILES}; do ${PYTHON} -m selectrl.checker.build --problems "${F}" --cache cache; done
${PYTHON} -m selectrl.checker.manifest write
log "checker cache ready under cache/checkers"

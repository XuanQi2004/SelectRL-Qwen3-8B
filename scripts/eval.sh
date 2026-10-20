#!/usr/bin/env bash
# Evaluate the base model or a checkpoint of a run with the locked protocol (inference only: all GPUs serve).
#
#   bash scripts/eval.sh base   [--split val|test|both] [--name frozen_base] [--protocol FILE] [--gpus 0,1]
#   bash scripts/eval.sh runs/<run> [--ckpt step_0050|final] [--split ...]
#
# Output: runs/<name>/eval/<split>_<ckpt>.{jsonl,summary.json,samples.jsonl.gz}; finished problems are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -ge 1 && "$1" != "-h" && "$1" != "--help" ]] || { sed -n '2,8p' "$0"; exit 0; }
TARGET="$1"; shift
SPLIT=both; CKPT=final; NAME=frozen_base; PROTOCOL="configs/protocol/qwen3_8b_40960.yaml"; GPUS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2;; --ckpt) CKPT="$2"; shift 2;; --name) NAME="$2"; shift 2;;
    --protocol) PROTOCOL="$2"; shift 2;; --gpus) GPUS="$2"; shift 2;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
export SELECTRL_PROTOCOL="${PROTOCOL}"
if [[ "${TARGET}" == base ]]; then
  RUN="runs/${NAME}"; ADAPTER=none; TAG=final
  mkdir -p "${RUN}"
  [[ -f "${RUN}/manifest.json" ]] || ${PYTHON} -c "
import json; from selectrl.config import load_protocol, protocol_hash
c = load_protocol('${PROTOCOL}')
json.dump({'run_name': '${NAME}', 'algo_name': 'base (no training)', 'protocol': c['protocol'],
           'protocol_hash': protocol_hash(c), 'model': c['model']['hf_id']}, open('${RUN}/manifest.json', 'w'), indent=1)"
else
  RUN="${TARGET%/}"
  if [[ "${CKPT}" == final ]]; then
    TOTAL=$(${PYTHON} -c "from selectrl.config import load_protocol; print(load_protocol('${PROTOCOL}')['train']['steps'])")
    ADAPTER="${RUN}/ckpt/step_$(printf '%04d' "${TOTAL}")/adapter"; TAG=final
  else
    ADAPTER="${RUN}/ckpt/${CKPT}/adapter"; TAG="${CKPT}"
  fi
  [[ -d "${ADAPTER}" ]] || { echo "adapter ${ADAPTER} not found" >&2; exit 1; }
fi
preset "${GPUS}"
ALL_GPUS=$(${PYTHON} -c "import os; print(','.join(sorted(set('${SELECTRL_SERVER_GPUS},${SELECTRL_TRAIN_GPUS}'.split(',')), key=int)))")
trap 'stop_servers' EXIT
mkdir -p "${RUN}/logs" "${RUN}/eval"
SPLITS=$([[ "${SPLIT}" == both ]] && echo "val test" || echo "${SPLIT}")
STARTED=0
for S in ${SPLITS}; do
  OUT="${RUN}/eval/${S}_${TAG}"
  [[ -f "${OUT}.summary.json" ]] && { log "${OUT} already evaluated"; continue; }
  if [[ "${STARTED}" == 0 ]]; then
    start_servers "${RUN}/servers_eval.json" "${RUN}/logs" "${ALL_GPUS}" "${SELECTRL_EVAL_MEM_UTIL:-0.85}" "${SELECTRL_MAX_NUM_SEQS}"
    STARTED=1
  fi
  ${PYTHON} -m selectrl.eval.run_eval --protocol "${PROTOCOL}" --split "${S}" --servers "${RUN}/servers_eval.json" \
    --adapter "${ADAPTER}" --out "${OUT}" 2>&1 | tee -a "${RUN}/logs/eval.log"
done
log "done: ${RUN}/eval"

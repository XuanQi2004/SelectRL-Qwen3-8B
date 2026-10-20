#!/usr/bin/env bash
# Train one algorithm with the locked protocol, then evaluate the final checkpoint on val and test.
#
#   bash scripts/train.sh <algorithm_dir> [--seed 0] [--name RUN_NAME] [--protocol configs/protocol/qwen3_8b_40960.yaml]
#                         [--gpus 0,1,2,3] [--no-eval]
#
# Output: runs/<RUN_NAME>/ (default RUN_NAME = <algorithm>_seed<seed>). Re-running the same command after an
# interruption resumes from the last finished step (and skips finished evaluations).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -ge 1 && "$1" != "-h" && "$1" != "--help" ]] || { sed -n '2,9p' "$0"; exit 0; }
ALGO="$1"; shift
SEED=0; NAME=""; PROTOCOL="configs/protocol/qwen3_8b_40960.yaml"; GPUS=""; DO_EVAL=1; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;; --name) NAME="$2"; shift 2;; --protocol) PROTOCOL="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;; --no-eval) DO_EVAL=0; shift;; --dry-run) DRY=1; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
export SELECTRL_PROTOCOL="${PROTOCOL}"
NAME="${NAME:-$(basename "${ALGO}")_seed${SEED}}"
RUN="runs/${NAME}"
${PYTHON} -m selectrl.bench.guard "${ALGO}"
preset "${GPUS}"
NTRAIN=$(awk -F, '{print NF}' <<< "${SELECTRL_TRAIN_GPUS}")
if [[ "${DRY}" == 1 ]]; then log "dry run: would train ${ALGO} into ${RUN} on ${NTRAIN} rank(s)"; exit 0; fi
mkdir -p "${RUN}/logs"
trap 'stop_servers' EXIT

TOTAL=$(${PYTHON} -c "from selectrl.config import load_protocol; print(load_protocol('${PROTOCOL}')['train']['steps'])")
for attempt in 1 2 3 4; do
  [[ -f "${RUN}/done" ]] && break
  start_servers "${RUN}/servers.json" "${RUN}/logs" "${SELECTRL_SERVER_GPUS}" "${SELECTRL_MEM_UTIL}" "${SELECTRL_MAX_NUM_SEQS}"
  log "training ${ALGO} -> ${RUN} (attempt ${attempt}, ${NTRAIN} training rank(s))"
  set +e
  CUDA_VISIBLE_DEVICES="${SELECTRL_TRAIN_GPUS}" ${PYTHON} -m torch.distributed.run --standalone \
    --nproc_per_node="${NTRAIN}" -m selectrl.train.main --run-dir "${RUN}" --algo "${ALGO}" \
    --protocol "${PROTOCOL}" --servers "${RUN}/servers.json" --seed "${SEED}" \
    --micro-tokens "${SELECTRL_MICRO_TOKENS}" 2>&1 | tee -a "${RUN}/logs/train.log"
  rc=${PIPESTATUS[0]}
  set -e
  unset SELECTRL_STOP_AFTER_STEP   # only the first attempt may stop early (resume test of scripts/smoke.sh)
  if [[ ${rc} -eq 0 && -f "${RUN}/done" ]]; then break; fi
  log "training exited with code ${rc}; restarting from the last finished step"
  stop_servers
  sleep 10
done
[[ -f "${RUN}/done" ]] || { log "ERROR: training did not finish after 4 attempts; see ${RUN}/logs/train.log"; exit 1; }

if [[ "${DO_EVAL}" == 1 ]]; then
  FINAL="${RUN}/ckpt/step_$(printf '%04d' "${TOTAL}")/adapter"
  for SPLIT in val test; do
    OUT="${RUN}/eval/${SPLIT}_final"
    [[ -f "${OUT}.summary.json" ]] && { log "${OUT} already evaluated"; continue; }
    if ! ${PYTHON} -c "from selectrl.config import load_protocol, resolve; import sys; sys.exit(0 if resolve(load_protocol('${PROTOCOL}')['data']['${SPLIT}']).exists() else 1)"; then
      log "split ${SPLIT} not present (agent workspace?): skipped"; continue
    fi
    [[ -n "${SERVERS_PID:-}" ]] || start_servers "${RUN}/servers.json" "${RUN}/logs" "${SELECTRL_SERVER_GPUS}" "${SELECTRL_MEM_UTIL}" "${SELECTRL_MAX_NUM_SEQS}"
    ${PYTHON} -m selectrl.eval.run_eval --protocol "${PROTOCOL}" --split "${SPLIT}" --servers "${RUN}/servers.json" \
      --adapter "${FINAL}" --out "${OUT}" 2>&1 | tee -a "${RUN}/logs/eval.log"
  done
fi
log "done: ${RUN}"

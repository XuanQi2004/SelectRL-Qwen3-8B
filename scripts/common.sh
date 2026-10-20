#!/usr/bin/env bash
# Shared settings for every script (sourced, not executed). All paths are relative to the repository root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-python}"
if [[ -z "${SELECTRL_NO_VENV:-}" && -x "${REPO_ROOT}/.venv/bin/python" && "${PYTHON}" == "python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi
export PYTHON
# local servers must never be reached through an HTTP proxy
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}" NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SELECTRL_MODEL_PATH="${SELECTRL_MODEL_PATH:-}"
mkdir -p logs runs

log() { echo "[$(date '+%F %T')] $*"; }

# start_servers <state_file> <log_dir> <gpu_list> <mem_util> <max_num_seqs>
# Starts the launcher in the background (one vLLM server per GPU) and waits until all servers are ready.
start_servers() {
  local state="$1" logdir="$2" gpus="$3" util="$4" seqs="$5"
  mkdir -p "${logdir}"
  rm -f "${state}"
  local model; model="$(${PYTHON} -c 'from selectrl.config import load_protocol, model_path; import os; print(model_path(load_protocol(os.environ.get("SELECTRL_PROTOCOL", "configs/protocol/qwen3_8b_40960.yaml"))))')"
  ${PYTHON} -m selectrl.rollout.servers launch --gpus "${gpus}" --model "${model}" --state "${state}" \
    --log-dir "${logdir}" --mem-util "${util}" --max-num-seqs "${seqs}" \
    --max-model-len "$(${PYTHON} -c 'from selectrl.config import load_protocol; import os; print(load_protocol(os.environ.get("SELECTRL_PROTOCOL", "configs/protocol/qwen3_8b_40960.yaml"))["model"]["max_model_len"])')" \
    >> "${logdir}/launcher.log" 2>&1 &
  SERVERS_PID=$!
  log "vLLM launcher pid ${SERVERS_PID} (GPUs ${gpus}, gpu-memory-utilization ${util}); logs in ${logdir}"
  if ! ${PYTHON} -m selectrl.rollout.servers wait --state "${state}" --pid "${SERVERS_PID}" --timeout 3600 > /dev/null; then
    log "ERROR: vLLM servers did not start. Last launcher lines:"; tail -n 60 "${logdir}/launcher.log" >&2
    stop_servers; return 3
  fi
  log "vLLM servers ready"
}

stop_servers() {
  if [[ -n "${SERVERS_PID:-}" ]] && kill -0 "${SERVERS_PID}" 2>/dev/null; then
    kill -TERM "${SERVERS_PID}" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 "${SERVERS_PID}" 2>/dev/null || break; sleep 1; done
    kill -KILL "${SERVERS_PID}" 2>/dev/null || true
  fi
  SERVERS_PID=""
}

# preset [gpu_list] -> exports SELECTRL_PLACEMENT, SELECTRL_SERVER_GPUS, SELECTRL_TRAIN_GPUS, SELECTRL_MEM_UTIL ...
preset() {
  eval "$(${PYTHON} -m selectrl.config preset ${1:+--gpus "$1"})"
  export SELECTRL_PLACEMENT SELECTRL_SERVER_GPUS SELECTRL_TRAIN_GPUS SELECTRL_MEM_UTIL SELECTRL_MAX_NUM_SEQS \
    SELECTRL_MICRO_TOKENS SELECTRL_NUM_GPUS SELECTRL_GPU_MEM_GB
  log "hardware: ${SELECTRL_NUM_GPUS} GPU(s) x ${SELECTRL_GPU_MEM_GB} GB, placement ${SELECTRL_PLACEMENT}, servers on" \
      "[${SELECTRL_SERVER_GPUS}] (util ${SELECTRL_MEM_UTIL}), trainers on [${SELECTRL_TRAIN_GPUS}]," \
      "micro-batch ${SELECTRL_MICRO_TOKENS} tokens"
}

#!/usr/bin/env bash
# Download the released reference runs (LoRA adapters, run manifests, per-step metrics, per-problem evaluation rows)
# into results/baselines/<run>/ and verify their sha256. Skips archives that are already installed.
#
#   bash scripts/fetch_reference.sh            # every released reference run
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,6p' "$0"; exit 0; }
BASE_URL="${SELECTRL_REFERENCE_URL:-https://github.com/THUQiXuan/SelectRL-Qwen3-8B/releases/download/v1.0.0}"
SHA_FILE="bench/REFERENCE.sha256"
[[ -f "${SHA_FILE}" ]] || { log "ERROR: ${SHA_FILE} missing"; exit 1; }
mkdir -p .downloads results/baselines
while read -r sha name; do
  [[ -n "${name}" ]] || continue
  if [[ -f ".downloads/${name}.installed" ]]; then log "${name} already installed"; continue; fi
  if [[ ! -f ".downloads/${name}" ]]; then
    log "downloading ${BASE_URL}/${name}"
    curl -L --fail --retry 5 -o ".downloads/${name}.part" "${BASE_URL}/${name}"
    mv ".downloads/${name}.part" ".downloads/${name}"
  fi
  echo "${sha}  .downloads/${name}" | sha256sum -c -
  tar -xzf ".downloads/${name}" -C results/baselines
  touch ".downloads/${name}.installed"
done < "${SHA_FILE}"
log "reference runs installed under results/baselines/"

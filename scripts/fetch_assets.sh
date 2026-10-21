#!/usr/bin/env bash
# Download the released data + checker cache (problems, splits, helper programs, check cases) from the GitHub
# release, verify their sha256 (bench/ASSETS.sha256) and the per-file lock (bench/LOCK.json). Skips the download
# when data/ and cache/ already match the lock.
#
#   bash scripts/fetch_assets.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,7p' "$0"; exit 0; }
BASE_URL="${SELECTRL_ASSET_URL:-https://github.com/XuanQi2004/SelectRL-Qwen3-8B/releases/download/v1.0.0}"
SHA_FILE="bench/ASSETS.sha256"
if ${PYTHON} -m selectrl.bench.integrity verify >/dev/null 2>&1; then log "data and checker cache already present and verified"; exit 0; fi
mkdir -p .downloads
while read -r sha name; do
  [[ -n "${name}" ]] || continue
  if [[ ! -f ".downloads/${name}" ]]; then
    log "downloading ${BASE_URL}/${name}"
    curl -L --fail --retry 5 -o ".downloads/${name}.part" "${BASE_URL}/${name}"
    mv ".downloads/${name}.part" ".downloads/${name}"
  fi
  echo "${sha}  .downloads/${name}" | sha256sum -c -
done < "${SHA_FILE}"
tar -xzf .downloads/selectrl-data-v1.tar.gz
cat .downloads/selectrl-checkers-v1.tar.gz.part* | tar -xz
${PYTHON} -m selectrl.bench.integrity verify
log "assets installed (data/, cache/checkers/)"

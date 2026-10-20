#!/usr/bin/env bash
# Download the released data + checker cache (problems, splits, helper programs, check cases) from the GitHub
# release and verify their sha256. Skips the download when data/ and cache/ already match the lock.
#
#   bash scripts/fetch_assets.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,6p' "$0"; exit 0; }
ASSET_URL="${SELECTRL_ASSET_URL:-https://github.com/THUQiXuan/SelectRL-Qwen3-8B/releases/download/v1.0.0/selectrl-assets-v1.tar.gz}"
SHA_FILE="bench/ASSETS.sha256"
if ${PYTHON} -m selectrl.bench.integrity verify >/dev/null 2>&1; then log "data and checker cache already present and verified"; exit 0; fi
mkdir -p .downloads
if [[ ! -f .downloads/assets.tar.gz ]]; then
  log "downloading ${ASSET_URL}"
  curl -L --fail --retry 5 -o .downloads/assets.tar.gz.part "${ASSET_URL}"
  mv .downloads/assets.tar.gz.part .downloads/assets.tar.gz
fi
if [[ -f "${SHA_FILE}" ]]; then
  echo "$(cat "${SHA_FILE}")  .downloads/assets.tar.gz" | sha256sum -c -
fi
tar -xzf .downloads/assets.tar.gz
${PYTHON} -m selectrl.bench.integrity verify
log "assets installed (data/, cache/checkers/)"

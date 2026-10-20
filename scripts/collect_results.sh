#!/usr/bin/env bash
# Pack the small result files of every run (no checkpoints, no model weights) into results_bundle.tar.gz.
#
#   bash scripts/collect_results.sh [--with-samples]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
WITH_SAMPLES=0; [[ "${1:-}" == "--with-samples" ]] && WITH_SAMPLES=1
LIST=$(mktemp)
find runs results -type f \( -name 'manifest.json' -o -name 'metrics.jsonl' -o -name 'done' -o -name 'score.json' \
  -o -name '*.summary.json' -o -path '*/eval/*.jsonl' -o -name '*.md' -o -name 'train.log' -o -name 'eval.log' \
  -o -name 'launcher.log' \) 2>/dev/null > "${LIST}"
if [[ "${WITH_SAMPLES}" == 1 ]]; then find runs -type f \( -name '*.samples.jsonl.gz' -o -name 'samples.jsonl.gz' \) >> "${LIST}"; fi
tar -czf results_bundle.tar.gz -T "${LIST}" logs 2>/dev/null || tar -czf results_bundle.tar.gz -T "${LIST}"
rm -f "${LIST}"
log "wrote results_bundle.tar.gz ($(du -h results_bundle.tar.gz | cut -f1)); send this file back"

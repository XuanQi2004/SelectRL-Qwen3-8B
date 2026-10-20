#!/usr/bin/env bash
# MAINTAINERS ONLY - the released data already contains everything this script produces.
# Rebuild the locked data from scratch: CodeContests download + conversion + contest-disjoint splits, active
# training pool (base pass rate strictly between 0 and 1 over 8 samples), then the checker cache for every split.
#
#   bash scripts/build_data.sh [--max-candidates 4000] [--gpus 0,1,...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
MAXC=4000; GPUS=""
while [[ $# -gt 0 ]]; do
  case "$1" in --max-candidates) MAXC="$2"; shift 2;; --gpus) GPUS="$2"; shift 2;;
    -h|--help) sed -n '2,7p' "$0"; exit 0;; *) echo "unknown argument $1" >&2; exit 2;; esac
done
[[ -f data/problems/train_candidates.jsonl.gz ]] || ${PYTHON} -m selectrl.data.codecontests --out data
preset "${GPUS}"
ALL=$(${PYTHON} -c "print(','.join(sorted(set('${SELECTRL_SERVER_GPUS},${SELECTRL_TRAIN_GPUS}'.split(',')), key=int)))")
trap 'stop_servers' EXIT
start_servers runs/_build/servers.json runs/_build/logs "${ALL}" 0.85 "${SELECTRL_MAX_NUM_SEQS}"
[[ -f data/problems/train.jsonl.gz ]] || ${PYTHON} -m selectrl.data.active --servers runs/_build/servers.json \
  --candidates data/problems/train_candidates.jsonl.gz --max-candidates "${MAXC}" --n 8 --out data/problems/train.jsonl.gz
${PYTHON} -m selectrl.checker.gen --servers runs/_build/servers.json \
  --problems data/problems/train.jsonl.gz data/problems/val.jsonl.gz data/problems/test.jsonl.gz
stop_servers
for S in train val test; do ${PYTHON} -m selectrl.checker.build --problems "data/problems/${S}.jsonl.gz" --cache cache; done
${PYTHON} -m selectrl.checker.manifest write
log "data rebuilt; refresh the lock with: ${PYTHON} -m selectrl.bench.integrity write"

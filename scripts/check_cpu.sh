#!/usr/bin/env bash
# CPU-only checks (no GPU, no network): unit tests of the harness and reference algorithms, the guard on every
# shipped algorithm, bash syntax of every script, and the integrity lock.
#
#   bash scripts/check_cpu.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,6p' "$0"; exit 0; }
for f in scripts/*.sh; do bash -n "$f"; done
for d in reference_algorithms/* algorithms/*; do [[ -f "$d/algorithm.py" ]] && ${PYTHON} -m selectrl.bench.guard "$d"; done
${PYTHON} -m pytest -q tests/
if [[ -f bench/LOCK.json ]]; then ${PYTHON} -m selectrl.bench.integrity verify; fi
log "CPU checks passed"

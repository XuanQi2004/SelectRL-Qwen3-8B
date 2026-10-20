#!/usr/bin/env bash
# Create a copy of this repository for a benchmark agent (hidden test data removed, everything except
# algorithms/ read-only) and print the task statement location.
#
#   bash scripts/make_agent_workspace.sh <out_dir>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 1 && "$1" != "-h" ]] || { sed -n '2,6p' "$0"; exit 0; }
${PYTHON} -m selectrl.bench.workspace make --out "$1"
log "give the agent: $1/AGENT_TASK.md ; collect its work with: ${PYTHON} -m selectrl.bench.workspace collect --workspace $1 --name <name>"

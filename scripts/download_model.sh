#!/usr/bin/env bash
# Download Qwen/Qwen3-8B at the pinned revision into models/Qwen3-8B (~16 GB).
# Uses HF_TOKEN if set (the model is public; a token only avoids anonymous rate limits). HF_ENDPOINT is respected.
#
#   bash scripts/download_model.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,6p' "$0"; exit 0; }
if [[ -z "${HF_TOKEN:-}" && -f .hf_token ]]; then export HF_TOKEN="$(tr -d ' \n' < .hf_token)"; fi
${PYTHON} - <<'PY'
import os
from huggingface_hub import snapshot_download
from selectrl.config import load_protocol, resolve
m = load_protocol("configs/protocol/qwen3_8b_40960.yaml")["model"]
dst = resolve(m["local_dir"])
if (dst / "config.json").exists() and len(list(dst.glob("*.safetensors"))) >= 4:
    print(f"{dst} already present"); raise SystemExit(0)
snapshot_download(repo_id=m["hf_id"], revision=m["revision"], local_dir=str(dst), token=os.environ.get("HF_TOKEN") or None,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "merges.txt", "vocab.json"])
print(f"downloaded {m['hf_id']}@{m['revision'][:8]} -> {dst}")
PY

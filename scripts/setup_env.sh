#!/usr/bin/env bash
# Install / verify the Python environment.
#
#   bash scripts/setup_env.sh              # add the few extra packages to the CURRENT python, which must already
#                                          # have vLLM 0.29.x (e.g. the vllm/vllm-openai:v0.29.0 image)
#   bash scripts/setup_env.sh --new-venv   # create .venv (uv if available) with vllm==0.29.0 + extras
#   bash scripts/setup_env.sh --check      # only print versions and run the checks
#
# torch / vllm / transformers / triton already installed are pinned by a constraints file, so this never
# upgrades or downgrades them.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
MODE=current
case "${1:-}" in --new-venv) MODE=venv;; --check) MODE=check;; -h|--help) sed -n '2,11p' "$0"; exit 0;; "") ;;
  *) echo "unknown argument $1" >&2; exit 2;; esac

if [[ "${MODE}" == venv ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python "vllm==0.29.0"
  else
    python3 -m venv .venv
    .venv/bin/python -m pip install -U pip
    .venv/bin/python -m pip install "vllm==0.29.0"
  fi
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

if [[ "${MODE}" != check ]]; then
  ${PYTHON} -c "import vllm, sys; v = vllm.__version__; print('vLLM', v); sys.exit(0 if v.startswith('0.29') else 1)" || {
    echo "ERROR: this python does not have vLLM 0.29.x (use --new-venv, or activate the vLLM 0.29 environment)" >&2; exit 1; }
  CONS=$(mktemp)
  ${PYTHON} -m pip freeze 2>/dev/null | grep -i -E '^(torch|torchvision|torchaudio|vllm|triton|xformers|flashinfer[a-z_-]*|numpy)==' > "${CONS}" || true
  if ! ${PYTHON} -c "import transformers.models.qwen3_5" 2>/dev/null; then
    log "installed transformers has no Qwen3.5 support -> installing transformers==5.12.1"
    grep -v -i '^transformers==' "${CONS}" > "${CONS}.2" && mv "${CONS}.2" "${CONS}"
    ${PYTHON} -m pip install -c "${CONS}" "transformers==5.12.1"
  fi
  ${PYTHON} -m pip install -c "${CONS}" -r requirements/extra.txt
  ${PYTHON} -m pip install --no-deps -e .
  rm -f "${CONS}"
fi

${PYTHON} - <<'PY'
import importlib, shutil, subprocess, sys
mods = ["vllm", "torch", "transformers", "peft", "fla", "aiohttp", "yaml", "datasets", "safetensors", "huggingface_hub"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        bad.append(m); print(f"  {m:16s} MISSING ({e.__class__.__name__}: {e})")
import torch
print(f"  CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(f"    GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GiB")
try:
    import transformers.models.qwen3_5  # noqa: F401
    print("  transformers Qwen3.5 support: yes")
except Exception:
    bad.append("transformers(qwen3_5)"); print("  transformers Qwen3.5 support: NO")
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    print("  fla gated-delta-rule kernels: yes")
except Exception as e:  # noqa: BLE001
    bad.append("fla kernels"); print(f"  fla gated-delta-rule kernels: NO ({e})")
un = shutil.which("unshare") and subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0
print(f"  network isolation for the sandbox (unshare -rn): {'available' if un else 'not available (Python-level socket block only)'}")
sys.exit(1 if bad else 0)
PY
log "environment OK"

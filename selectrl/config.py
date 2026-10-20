"""Protocol loading (with `inherit:`), protocol hashing and hardware presets.

    python -m selectrl.config show  --protocol configs/protocol/qwen3_8b_40960.yaml
    python -m selectrl.config preset [--gpus 0,1,2,3]     # prints shell assignments used by scripts/*.sh

Hardware presets (single node, 1-8 GPUs, every GPU gets one vLLM server unless placement is split):

  GPU memory >= 120 GB (H200, B200 ...)  colocate: server + training rank on every GPU, vLLM gets 40% of memory
  GPU memory 75-120 GB (H100/A100 80 GB) colocate with a smaller KV budget (42%) and smaller micro-batches
  GPU memory < 75 GB (e.g. 48 GB)        split: first half of the GPUs serve, the second half train (>= 2 GPUs)

Override anything with environment variables: SELECTRL_PLACEMENT=colocate|split, SELECTRL_MEM_UTIL,
SELECTRL_MAX_NUM_SEQS, SELECTRL_MICRO_TOKENS, SELECTRL_SERVER_GPUS, SELECTRL_TRAIN_GPUS.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_protocol(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    cfg = yaml.safe_load(p.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load_protocol(p.parent / parent), cfg)
    return cfg


def protocol_hash(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def model_path(cfg: Dict[str, Any]) -> str:
    """Local model directory if downloaded (models/...), else the HF id (works when HF_HOME already has it)."""
    env = os.environ.get("SELECTRL_MODEL_PATH")
    if env:
        return env
    local = resolve(cfg["model"]["local_dir"])
    if (local / "config.json").exists():
        return str(local)
    return cfg["model"]["hf_id"]


# ---------------------------------------------------------------------------------------------- hardware
def detect_gpus() -> List[Dict[str, Any]]:
    """[{index, name, mem_gb}] for the GPUs visible to this process (CUDA_VISIBLE_DEVICES respected)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, mem = [x.strip() for x in line.split(",")]
        gpus.append({"index": idx, "name": name, "mem_gb": float(mem) / 1024.0})
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if vis:
        # child processes address GPUs by the ids in CUDA_VISIBLE_DEVICES; under a job scheduler nvidia-smi may
        # number the same devices differently, so keep the visible ids and take the memory from nvidia-smi
        sel = [g for g in gpus if g["index"] in vis]
        mem = min((g["mem_gb"] for g in (sel or gpus)), default=0.0)
        name = (sel or gpus or [{"name": "?"}])[0]["name"]
        return [{"index": v, "name": name, "mem_gb": mem} for v in vis]
    return gpus


def hardware_preset(gpu_indices: Optional[List[str]] = None, mem_gb: Optional[float] = None) -> Dict[str, Any]:
    gpus = detect_gpus()
    if gpu_indices is None:
        gpu_indices = [g["index"] for g in gpus]
    if mem_gb is None:
        mem_gb = min((g["mem_gb"] for g in gpus if g["index"] in gpu_indices), default=0.0)
    n = len(gpu_indices)
    if n == 0:
        raise RuntimeError("no GPU found (nvidia-smi); this harness needs at least one NVIDIA GPU")
    placement = os.environ.get("SELECTRL_PLACEMENT") or ("colocate" if mem_gb >= 75 else "split")
    if placement == "split" and n < 2:
        raise RuntimeError(f"{n} GPU with {mem_gb:.0f} GB: the policy needs >= 75 GB per GPU when one GPU must "
                           "hold both the inference server and the trainer; use >= 2 GPUs (split placement)")
    if placement == "colocate":
        server_gpus = train_gpus = list(gpu_indices)
        if mem_gb >= 120:
            mem_util, max_num_seqs, micro_tokens = 0.40, 256, 32768
        else:
            mem_util, max_num_seqs, micro_tokens = 0.42, 128, 12288
    else:
        half = max(1, n // 2)
        server_gpus, train_gpus = list(gpu_indices[:half]), list(gpu_indices[half:])
        mem_util = 0.88
        max_num_seqs = 256 if mem_gb >= 75 else 128
        micro_tokens = 32768 if mem_gb >= 120 else (16384 if mem_gb >= 75 else 8192)
    env = os.environ
    preset = {
        "placement": placement,
        "gpu_mem_gb": round(mem_gb, 1),
        "num_gpus": n,
        "server_gpus": env.get("SELECTRL_SERVER_GPUS", ",".join(server_gpus)).split(","),
        "train_gpus": env.get("SELECTRL_TRAIN_GPUS", ",".join(train_gpus)).split(","),
        "mem_util": float(env.get("SELECTRL_MEM_UTIL", mem_util)),
        "max_num_seqs": int(env.get("SELECTRL_MAX_NUM_SEQS", max_num_seqs)),
        "micro_tokens": int(env.get("SELECTRL_MICRO_TOKENS", micro_tokens)),
    }
    return preset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--protocol", default="configs/protocol/qwen3_8b_40960.yaml")
    p = sub.add_parser("preset")
    p.add_argument("--gpus", default=None, help="comma-separated GPU indices (default: all visible)")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.cmd == "show":
        cfg = load_protocol(a.protocol)
        print(json.dumps({"hash": protocol_hash(cfg), "protocol": cfg}, indent=2))
        return
    pr = hardware_preset(a.gpus.split(",") if a.gpus else None)
    if a.json:
        print(json.dumps(pr, indent=2))
        return
    for k, v in pr.items():
        v = ",".join(v) if isinstance(v, list) else v
        print(f"SELECTRL_{k.upper()}={shlex.quote(str(v))}")


if __name__ == "__main__":
    main()

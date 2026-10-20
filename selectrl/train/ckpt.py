"""Atomic checkpoints for exact resume.

Layout of a run directory (all paths relative to it):

    resume/            latest resumable state, rewritten after every step (adapter + optimizer + trainer state)
    ckpt/step_XXXX/    kept adapters every `save_every` steps and at the final step (used for evaluation)
    adapters/step_XXXX the adapter the vLLM servers serve for the next step's rollouts (last two kept)
    steps/step_XXXX/   rollout batch of a step (reused when a crash happens between sampling and the update)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _replace_dir(tmp: Path, final: Path) -> None:
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)


def export_for_vllm(peft_model, out_dir: Path, key_prefix_map: Dict[str, str], target_names=None) -> Path:
    """Save the adapter in the layout the vLLM server applies (bf16 weights, module names of the HF checkpoint).

    The trainer loads the text-only ``Qwen3_5ForCausalLM`` (modules ``model.layers.*``) while vLLM serves the
    checkpoint's ``Qwen3_5ForConditionalGeneration`` (modules ``model.language_model.layers.*``). vLLM accepts an
    adapter with the text-only names but silently applies nothing, so the keys are rewritten here.
    """
    from safetensors.torch import load_file, save_file

    tmp = out_dir.with_name(out_dir.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    peft_model.save_pretrained(str(tmp), safe_serialization=True)
    if key_prefix_map:
        f = tmp / "adapter_model.safetensors"
        sd = load_file(str(f))
        out = {}
        for k, v in sd.items():
            for a, b in key_prefix_map.items():
                if k.startswith(a):
                    k = b + k[len(a):]
                    break
            out[k] = v.to(torch.bfloat16)
        save_file(out, str(f))
    if target_names:  # plain module names (the layout verified with vLLM 0.29), not the trainer's regex
        cf = tmp / "adapter_config.json"
        cfg = json.loads(cf.read_text())
        cfg["target_modules"] = sorted(target_names)
        cf.write_text(json.dumps(cfg, indent=2))
    _replace_dir(tmp, out_dir)
    return out_dir


def save_resume(run_dir: Path, peft_model, optimizer, state: Dict[str, Any], vllm_export: Optional[Path] = None) -> None:
    """Write resume/ atomically: adapter/ (trainer layout), adapter_vllm/ (served layout), optimizer.pt, state.json."""
    final = run_dir / "resume"
    tmp = run_dir / f"resume.tmp{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    peft_model.save_pretrained(str(tmp / "adapter"), safe_serialization=True)
    if vllm_export is not None:
        shutil.copytree(vllm_export, tmp / "adapter_vllm")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    state = dict(state, saved_at=time.time())
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    _replace_dir(tmp, final)


def load_resume(run_dir: Path) -> Optional[Dict[str, Any]]:
    st = run_dir / "resume" / "state.json"
    if not st.exists():
        return None
    return json.loads(st.read_text())


def restore_weights(peft_model, optimizer, run_dir: Path, device) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    ad = run_dir / "resume" / "adapter" / "adapter_model.safetensors"
    sd = load_file(str(ad), device=str(device))
    res = set_peft_model_state_dict(peft_model, sd)
    unexpected = getattr(res, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"resume adapter has unexpected keys, e.g. {unexpected[:3]}")
    optimizer.load_state_dict(torch.load(run_dir / "resume" / "optimizer.pt", map_location=device,
                                         weights_only=False))


def prune_adapters(adapters_dir: Path, keep: int = 2) -> None:
    dirs = sorted(d for d in adapters_dir.glob("step_*") if d.is_dir() and "." not in d.name)
    for d in dirs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)

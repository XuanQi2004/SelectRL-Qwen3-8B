"""Policy model: causal LM (Qwen3-8B in the locked protocol) + PEFT LoRA, memory-safe token log-probabilities.

The vocabulary has ~248k entries, so logits are never materialized for a whole sequence: the backbone's hidden
states at response positions are gathered and pushed through the (frozen) LM head in small chunks, each chunk
wrapped in activation checkpointing so the backward pass recomputes its logits instead of storing them.
"""
from __future__ import annotations

import re
from contextlib import contextmanager, nullcontext
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LOGIT_CHUNK = 1024


def lora_target_regex(hf_model: torch.nn.Module, names: Sequence[str]) -> str:
    """Regex restricting LoRA to the text decoder layers (never the vision tower or the MTP head)."""
    alt = "|".join(re.escape(n) for n in names)
    mods = [n for n, _ in hf_model.named_modules()]
    if any(".language_model.layers." in n for n in mods):
        return rf".*language_model\.layers\.\d+\..*\.({alt})$"
    return rf"^model\.layers\.\d+\..*\.({alt})$"


def text_backbone(hf_model: torch.nn.Module) -> torch.nn.Module:
    """The decoder stack that maps input ids to final hidden states (after the last norm)."""
    inner = getattr(hf_model, "model", hf_model)
    return getattr(inner, "language_model", inner)


def lm_head_weight(hf_model: torch.nn.Module) -> torch.Tensor:
    return hf_model.get_output_embeddings().weight


def vllm_key_prefix_map(model_path: str, hf_model: torch.nn.Module) -> dict:
    """Adapter key rewrite needed when the checkpoint nests the text model under ``model.language_model``."""
    import json
    import os

    idx = os.path.join(model_path, "model.safetensors.index.json")
    nested = False
    if os.path.exists(idx):
        with open(idx) as f:
            nested = any(k.startswith("model.language_model.") for k in json.load(f)["weight_map"])
    else:
        try:
            from huggingface_hub import hf_hub_download

            with open(hf_hub_download(model_path, "model.safetensors.index.json")) as f:
                nested = any(k.startswith("model.language_model.") for k in json.load(f)["weight_map"])
        except Exception:  # noqa: BLE001 - single-file checkpoints have no index
            nested = False
    mods = [n for n, _ in hf_model.named_modules()]
    if nested and not any(".language_model." in n for n in mods):
        return {"base_model.model.model.layers.": "base_model.model.model.language_model.layers."}
    return {}


class PolicyModel:
    def __init__(self, model_path: str, lora_cfg: dict, device: torch.device, grad_ckpt: bool = True):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                  device_map={"": str(device)})
        hf.config.use_cache = False
        for p in hf.parameters():
            p.requires_grad_(False)
        if grad_ckpt:
            hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.target_regex = lora_target_regex(hf, lora_cfg["target_modules"])
        cfg = LoraConfig(r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]),
                         lora_dropout=float(lora_cfg.get("dropout", 0.0)), target_modules=self.target_regex,
                         bias="none", task_type="CAUSAL_LM")
        self.peft = get_peft_model(hf, cfg)
        self.peft.to(device)
        self.device = device
        self.hf = self.peft.base_model.model
        self.backbone = text_backbone(self.hf)
        self.head_w = lm_head_weight(self.hf)
        self.vllm_key_map = vllm_key_prefix_map(model_path, hf)
        self.trainable: List[torch.nn.Parameter] = [p for p in self.peft.parameters() if p.requires_grad]
        if not self.trainable:
            raise RuntimeError(f"LoRA matched no modules (regex {self.target_regex})")

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable)

    @contextmanager
    def base_only(self):
        """Temporarily disable the adapter (reference / frozen base model)."""
        with self.peft.disable_adapter():
            yield

    def train(self, mode: bool = True) -> None:
        self.peft.train(mode)

    # ------------------------------------------------------------------ log-probs
    def response_logprobs(self, batch: "MicroBatch", grad: bool) -> torch.Tensor:
        """Flat (T,) float32 log-probs of the response tokens of a right-padded micro-batch."""
        ctx = nullcontext() if grad else torch.no_grad()
        with ctx:
            hidden = self.backbone(input_ids=batch.input_ids, use_cache=False).last_hidden_state  # (B, L, H)
            h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, batch.pred_index)  # (T, H)
            return chunked_token_logprobs(h, batch.targets, self.head_w, grad)


def _chunk_lp(h: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h, w).float()
    return logits.gather(1, y[:, None])[:, 0] - torch.logsumexp(logits, dim=-1)


def chunked_token_logprobs(h: torch.Tensor, y: torch.Tensor, w: torch.Tensor, grad: bool,
                           chunk: int = LOGIT_CHUNK) -> torch.Tensor:
    outs = []
    for i in range(0, h.shape[0], chunk):
        hc, yc = h[i:i + chunk], y[i:i + chunk]
        if grad and hc.requires_grad:
            outs.append(checkpoint(_chunk_lp, hc, yc, w, use_reentrant=False))
        else:
            outs.append(_chunk_lp(hc, yc, w))
    return torch.cat(outs) if outs else h.new_zeros(0, dtype=torch.float32)


class MicroBatch:
    """Right-padded micro-batch of (prompt + response) sequences; only response tokens are scored."""

    def __init__(self, seqs: Sequence[Tuple[Sequence[int], Sequence[int]]], pad_id: int, device: torch.device):
        lens = [len(p) + len(r) for p, r in seqs]
        L = max(lens)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        pred, tgt, seq_index, token_pos, seq_len = [], [], [], [], []
        for b, (p, r) in enumerate(seqs):
            full = list(p) + list(r)
            ids[b, :len(full)] = torch.tensor(full, dtype=torch.long)
            P, R = len(p), len(r)
            pred.extend(b * L + P - 1 + j for j in range(R))
            tgt.extend(r)
            seq_index.extend([b] * R)
            token_pos.extend(range(R))
            seq_len.append(R)
        self.input_ids = ids.to(device, non_blocking=True)
        self.pred_index = torch.tensor(pred, dtype=torch.long, device=device)
        self.targets = torch.tensor(tgt, dtype=torch.long, device=device)
        self.seq_index = torch.tensor(seq_index, dtype=torch.long, device=device)
        self.token_pos = torch.tensor(token_pos, dtype=torch.long, device=device)
        self.seq_len = torch.tensor(seq_len, dtype=torch.long, device=device)
        self.num_tokens = len(tgt)
        self.padded_tokens = len(seqs) * L


def pack_micro_batches(lengths: Sequence[Tuple[int, int]], budget: int) -> List[List[int]]:
    """Group sequence indices (sorted by length) so that batch_size * max_len <= budget (one sequence minimum)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i][0] + lengths[i][1], reverse=True)
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_max = 0
    for i in order:
        L = lengths[i][0] + lengths[i][1]
        new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = L
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def balance_across_ranks(costs: Iterable[int], world: int) -> List[List[int]]:
    """Greedy longest-first assignment of items to ranks by cost (deterministic)."""
    items = sorted(enumerate(costs), key=lambda x: (-x[1], x[0]))
    loads = [0] * world
    out: List[List[int]] = [[] for _ in range(world)]
    for i, c in items:
        r = min(range(world), key=lambda j: (loads[j], j))
        out[r].append(i)
        loads[r] += c
    return out

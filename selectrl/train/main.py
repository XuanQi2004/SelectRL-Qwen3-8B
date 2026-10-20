"""RL training entry point (one process per training GPU, launched by torchrun from scripts/train.sh).

    torchrun --nproc_per_node G -m selectrl.train.main --run-dir runs/<name> --algo algorithms/candidate \
        --protocol configs/protocol/qwen3_8b_40960.yaml --servers runs/<name>/servers.json [--seed 0]

Rank 0 does all sampling, execution and the algorithm's advantage computation, and writes the step batch to
``runs/<name>/steps/step_XXXX/batch.pt``; every rank then trains on a token-balanced shard of it. Re-running the
same command after a crash resumes from ``runs/<name>/resume`` (state after the last finished step).
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from selectrl.algo_api import LOSS_AGG_MODES, Group, LossInputs, StepContext, validate_advantages
from selectrl.config import load_protocol, model_path, protocol_hash, resolve
from selectrl.utils import atomic_write_json, log

# ---------------------------------------------------------------------------------------------- helpers


def read_jsonl_gz(path: Path) -> List[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_cases(cache_root: Path, pid: str) -> dict:
    """Checker cases of a problem; a missing file is an error (the checker cache ships with the data)."""
    from selectrl.checker.build import load_cases as _load

    cases = _load(cache_root / "checkers" / "cases", pid)
    if cases is None:
        if os.environ.get("SELECTRL_ALLOW_MISSING_CASES") == "1":
            return {"pid": pid, "inputs": []}
        raise FileNotFoundError(f"no checker cases for {pid} under {cache_root}/checkers/cases "
                                "(run scripts/fetch_assets.sh or scripts/build_checkers.sh)")
    return cases


def stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def append_metrics(path: Path, row: dict) -> None:
    """Append one JSON row, replacing an earlier row of the same step (idempotent across resumes)."""
    rows = []
    if path.exists():
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("step") != row["step"]]
    rows.append(row)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, path)


class Dist:
    def __init__(self) -> None:
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.cpu_group = None
        if self.world > 1:
            dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6), device_id=self.device)
            self.cpu_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=6))

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier(group=self.cpu_group)

    def bcast(self, obj: Any) -> Any:
        if self.world == 1:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self.cpu_group)
        return box[0]

    def allreduce_(self, t: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(t)
        return t


# ---------------------------------------------------------------------------------------------- rank 0: sampling


class Rollouts:
    """Rank-0 side: problem schedule, sampling, execution, groups, algorithm call, step batch files."""

    def __init__(self, cfg: dict, run_dir: Path, servers_state: Path, seed: int, algo, state: dict):
        from selectrl.checker.cache import ExecCache
        from selectrl.rollout.client import ServerPool
        from selectrl.rollout.prompts import ChatRenderer

        self.cfg, self.run_dir, self.seed, self.algo = cfg, run_dir, seed, algo
        self.tr, self.sp = cfg["train"], cfg["sampling"]
        self.pool = ServerPool.from_state(servers_state)
        self.render = ChatRenderer(model_path(cfg), enable_thinking=bool(cfg["prompt"]["enable_thinking"]))
        self.problems = read_jsonl_gz(resolve(cfg["data"]["train"]))
        if not self.problems:
            raise RuntimeError("empty training split")
        self.by_pid = {p["pid"]: p for p in self.problems}
        self.cache_root = resolve("cache")
        self.exec = ExecCache(self.cache_root, cfg["checker"]["version"])
        self.sigma_plus: Dict[str, List[str]] = state.get("sigma_plus") or {}
        if not self.sigma_plus:
            self._init_sigma_plus()
        self.cursor = int(state.get("cursor", 0))
        self.epoch = int(state.get("epoch", 0))

    def _init_sigma_plus(self) -> None:
        """Known-correct signatures seeded from the Python reference solutions of the training problems."""
        items = []
        for p in self.problems:
            for code in (p.get("py_refs") or [])[:3]:
                items.append((p, load_cases(self.cache_root, p["pid"]), code))
        recs = self.exec.evaluate_many(items) if items else []
        sp: Dict[str, set] = {p["pid"]: set() for p in self.problems}
        for (p, _, _), r in zip(items, recs):
            if r["g"] == 1 and r["has_code"]:
                sp[p["pid"]].add(r["sig"])
        self.sigma_plus = {k: sorted(v) for k, v in sp.items()}
        log(f"[rollout] known-correct signatures seeded for {sum(1 for v in sp.values() if v)} problems")

    def _order(self, epoch: int) -> List[str]:
        rng = np.random.default_rng(stable_seed("order", self.tr["data_seed"], self.seed, epoch))
        pids = sorted(self.by_pid)
        return [pids[i] for i in rng.permutation(len(pids))]

    def next_problems(self) -> List[str]:
        k = int(self.tr["problems_per_step"])
        out: List[str] = []
        while len(out) < k:
            order = self._order(self.epoch)
            take = order[self.cursor:self.cursor + (k - len(out))]
            out += take
            self.cursor += len(take)
            if self.cursor >= len(order):
                self.epoch += 1
                self.cursor = 0
        return out

    def make_batch(self, step: int, adapter: Optional[str]) -> Dict[str, Any]:
        from selectrl.checker.chk import last_code_block as extract_program
        from selectrl.rollout.client import SampleRequest

        t0 = time.time()
        n = int(self.tr["rollouts_per_problem"])
        pids = self.next_problems()
        reqs = []
        for pid in pids:
            ids = list(self.render.solve_prompt_ids(self.by_pid[pid]["question"]))
            reqs.append(SampleRequest(ids, n, int(self.sp["max_new_tokens"]), float(self.sp["temperature"]),
                                      float(self.sp["top_p"]), int(self.sp["top_k"]),
                                      stable_seed("train", self.seed, step, pid), adapter, pid))
        outs = self.pool.sample_many(reqs)
        t_gen = time.time() - t0

        items, flat = [], []
        for req, choices in zip(reqs, outs):
            p = self.by_pid[req.tag]
            cases = load_cases(self.cache_root, req.tag)
            for c in choices:
                code = extract_program(c["text"])
                items.append((p, cases, code))
                flat.append((req, c, code))
        recs = self.exec.evaluate_many(items)
        t_exec = time.time() - t0 - t_gen

        groups: List[Group] = []
        per_group: List[List[Tuple[dict, dict]]] = []
        for gi, (req, choices) in enumerate(zip(reqs, outs)):
            rows = [(flat[gi * n + j][1], recs[gi * n + j]) for j in range(n)]
            per_group.append(rows)
            sig_ids: Dict[str, int] = {}
            known = set(self.sigma_plus.get(req.tag, []))
            sid, skc = [], []
            for _, r in rows:
                if not r["has_code"]:
                    sid.append(-1)
                    skc.append(0)
                    continue
                sid.append(sig_ids.setdefault(r["sig"], len(sig_ids)))
                skc.append(int(r["sig"] in known))
            arr = lambda xs, dt=np.int64: np.asarray(xs, dtype=dt)  # noqa: E731
            groups.append(Group(
                problem_key=req.tag,
                g=arr([r["g"] for _, r in rows]), chk=arr([r["chk"] for _, r in rows]),
                public_pass=arr([r["public_pass"] for _, r in rows]),
                public_frac=arr([r["public_frac"] for _, r in rows], np.float64),
                has_code=arr([r["has_code"] for _, r in rows]),
                truncated=arr([int(c["finish_reason"] == "length") for c, _ in rows]),
                sig_id=arr(sid), sig_known_correct=arr(skc),
                response_len=arr([len(c["token_ids"]) for c, _ in rows]),
            ))

        ctx = StepContext(step=step, total_steps=int(self.tr["steps"]), n_rollouts=n,
                          k_deploy=int(self.tr["k_deploy"]), max_new_tokens=int(self.sp["max_new_tokens"]),
                          seed=stable_seed("algo", self.seed, step))
        t1 = time.time()
        advs = validate_advantages(groups, call_with_timeout(self.algo.compute_advantages, groups, ctx,
                                                             timeout=float(self.tr.get("algo_timeout_s", 600))))
        t_algo = time.time() - t1
        for gi, rows in enumerate(per_group):  # Sigma^+ grows only after this step's advantages are fixed
            s = set(self.sigma_plus.get(groups[gi].problem_key, []))
            s |= {r["sig"] for _, r in rows if r["g"] == 1 and r["has_code"]}
            self.sigma_plus[groups[gi].problem_key] = sorted(s)

        drop_zero = bool(getattr(self.algo, "drop_zero_adv_groups", True))
        seqs, group_of = [], []
        kept_groups = 0
        for gi, (req, rows) in enumerate(zip(reqs, per_group)):
            a = advs[gi]
            if drop_zero and not np.any(a != 0.0):
                continue
            kept_groups += 1
            for j, (c, _) in enumerate(rows):
                if not c["token_ids"]:
                    continue
                seqs.append({"prompt": req.prompt_ids, "resp": c["token_ids"], "rlp": c["logprobs"],
                             "adv": float(a[j]), "group": gi})
                group_of.append(gi)
        # mini-batches: random split of groups (seeded); normalizers are global per mini-batch
        M = max(1, int(self.tr["ppo_mini_batches"]))
        gids = sorted(set(group_of))
        rng = random.Random(stable_seed("mb", self.seed, step))
        rng.shuffle(gids)
        part = {g: i % M for i, g in enumerate(gids)}
        mbs: List[List[int]] = [[] for _ in range(M)]
        for si, s in enumerate(seqs):
            mbs[part[s["group"]]].append(si)
        all_groups_per_mb = [len(reqs) / M] * M
        norms = []
        for m in range(M):
            idx = mbs[m]
            n_tok = sum(len(seqs[i]["resp"]) for i in idx)
            norms.append({
                "token_mean": max(1, n_tok),
                "seq_mean_token_mean": max(1, len(idx)),
                "seq_mean_token_sum_norm": max(1.0, all_groups_per_mb[m] * n * int(self.tr["loss_norm_tokens"])),
            })

        flat_recs = [r for rows in per_group for _, r in rows]
        flat_ch = [c for rows in per_group for c, _ in rows]
        all_adv = np.concatenate(advs) if advs else np.zeros(0)
        metrics = {
            "n_problems": len(reqs), "n_samples": len(flat_recs), "kept_groups": kept_groups,
            "kept_seqs": len(seqs),
            "g_rate": float(np.mean([r["g"] for r in flat_recs])),
            "chk_rate": float(np.mean([r["chk"] for r in flat_recs])),
            "public_pass_rate": float(np.mean([r["public_pass"] for r in flat_recs])),
            "has_code_rate": float(np.mean([r["has_code"] for r in flat_recs])),
            "truncated_rate": float(np.mean([c["finish_reason"] == "length" for c in flat_ch])),
            "resp_len_mean": float(np.mean([len(c["token_ids"]) for c in flat_ch])),
            "solved_any_rate": float(np.mean([g.g.max() for g in groups])),
            "adv_mean": float(all_adv.mean()) if all_adv.size else 0.0,
            "adv_std": float(all_adv.std()) if all_adv.size else 0.0,
            "adv_absmax": float(np.abs(all_adv).max()) if all_adv.size else 0.0,
            "t_generate_s": round(t_gen, 1), "t_exec_s": round(t_exec, 1), "t_algo_s": round(t_algo, 2),
            "exec_cache_hits": self.exec.hits, "exec_cache_misses": self.exec.misses,
        }
        metrics.update({f"algo/{k}": float(v) for k, v in (self.algo.metrics() or {}).items()})
        log_rows = [{"pid": r_.tag, "j": j, "len": len(c["token_ids"]), "finish": c["finish_reason"],
                     "g": rec["g"], "chk": rec["chk"], "sig": rec["sig"], "adv": float(advs[gi][j])}
                    for gi, (r_, rows) in enumerate(zip(reqs, per_group)) for j, (c, rec) in enumerate(rows)]
        return {"step": step, "seqs": seqs, "mini_batches": mbs, "norms": norms, "metrics": metrics,
                "pids": pids, "cursor": self.cursor, "epoch": self.epoch, "sigma_plus": self.sigma_plus,
                "algo_state": json.loads(json.dumps(self.algo.state)), "log_rows": log_rows,
                "texts": [c["text"] for c in flat_ch]}


def call_with_timeout(fn, *args, timeout: float):
    """Run fn(*args) in a worker thread; fail the step if it does not return within `timeout` seconds."""
    import concurrent.futures as cf

    ex = cf.ThreadPoolExecutor(1)
    fut = ex.submit(fn, *args)
    try:
        return fut.result(timeout=timeout)
    except cf.TimeoutError:
        raise RuntimeError(f"algorithm.compute_advantages exceeded {timeout:.0f}s") from None
    finally:
        ex.shutdown(wait=False)


# ---------------------------------------------------------------------------------------------- all ranks: update


def train_on_batch(batch: dict, model, optimizer, algo, d: Dist, micro_tokens: int, max_grad_norm: float,
                   pad_id: int) -> Dict[str, float]:
    from selectrl.train.model import MicroBatch, balance_across_ranks, pack_micro_batches

    seqs = batch["seqs"]
    mode = algo.loss_agg_mode
    if mode not in LOSS_AGG_MODES:
        raise ValueError(f"unknown loss_agg_mode {mode!r}")
    # rank assignment per mini-batch (identical on every rank)
    mine: List[List[int]] = []
    for idx in batch["mini_batches"]:
        costs = [len(seqs[i]["prompt"]) + len(seqs[i]["resp"]) for i in idx]
        shards = balance_across_ranks(costs, d.world)
        mine.append([idx[j] for j in shards[d.rank]])

    def micro(idx: List[int]) -> List[List[int]]:
        lens = [(len(seqs[i]["prompt"]), len(seqs[i]["resp"])) for i in idx]
        return [[idx[j] for j in mb] for mb in pack_micro_batches(lens, micro_tokens)]

    def mk(idx: List[int]) -> MicroBatch:
        return MicroBatch([(seqs[i]["prompt"], seqs[i]["resp"]) for i in idx], pad_id, d.device)

    # old / reference log-probs before any update of this step
    old_lp: Dict[int, torch.Tensor] = {}
    ref_lp: Dict[int, torch.Tensor] = {}
    model.train(False)
    need_old = len(batch["mini_batches"]) > 1
    for m_idx in mine:
        for mb in micro(m_idx):
            b = mk(mb)
            if need_old:
                lp = model.response_logprobs(b, grad=False)
                _scatter(lp, b, mb, old_lp)
            if algo.needs_ref_logp:
                with model.base_only():
                    lp = model.response_logprobs(b, grad=False)
                _scatter(lp, b, mb, ref_lp)
    model.train(True)

    stats = torch.zeros(10, device=d.device, dtype=torch.float64)
    for m, m_idx in enumerate(mine):
        norm = float(batch["norms"][m][mode])
        for mb in micro(m_idx):
            b = mk(mb)
            logp = model.response_logprobs(b, grad=True)
            cat = lambda dct: torch.cat([dct[i] for i in mb]) if dct else None  # noqa: E731
            old = cat(old_lp) if need_old else logp.detach()
            ref = cat(ref_lp) if algo.needs_ref_logp else None
            rlp = torch.tensor([x for i in mb for x in seqs[i]["rlp"]], device=d.device, dtype=torch.float32)
            adv = torch.tensor([seqs[i]["adv"] for i in mb], device=d.device, dtype=torch.float32)[b.seq_index]
            inp = LossInputs(logp=logp, old_logp=old, ref_logp=ref, rollout_logp=rlp, advantages=adv,
                             seq_index=b.seq_index, token_pos=b.token_pos, seq_len=b.seq_len)
            tok_loss = algo.policy_loss(inp)
            if not torch.is_tensor(tok_loss) or tok_loss.shape != logp.shape:
                raise ValueError(f"policy_loss must return a per-token tensor of shape {tuple(logp.shape)}")
            if mode == "seq_mean_token_mean":
                per_tok_w = 1.0 / b.seq_len[b.seq_index].float()
                loss = (tok_loss * per_tok_w).sum() / norm
            else:
                loss = tok_loss.sum() / norm
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss from policy_loss")
            loss.backward()
            with torch.no_grad():
                ratio = torch.exp(logp.detach() - old)
                pre = m == 0  # before the first optimizer update of the step: logp is the sampling policy
                stats += torch.tensor([loss.item(), b.num_tokens, (logp.detach() - rlp).abs().sum().item() if pre else 0.0,
                                       ((ratio - 1).abs() > 0.2).float().sum().item(),
                                       (rlp - logp.detach()).sum().item() if pre else 0.0, b.num_tokens if pre else 0, 0, 0,
                                       (logp.detach() - ref).abs().sum().item() if (pre and ref is not None) else 0.0,
                                       b.num_tokens if (pre and ref is not None) else 0],
                                      device=d.device, dtype=torch.float64)
        # all-reduce gradients (sum: losses are already divided by global normalizers)
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in model.trainable]
        flat = torch.cat([g.reshape(-1).float() for g in grads])
        d.allreduce_(flat)
        off = 0
        for p in model.trainable:
            k = p.numel()
            p.grad = flat[off:off + k].view_as(p).to(p.dtype)
            off += k
        gn = torch.nn.utils.clip_grad_norm_(model.trainable, max_grad_norm)
        stats[6] += float(gn)
        stats[7] += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    d.allreduce_(stats)
    ntok = max(1.0, stats[1].item())
    npre = max(1.0, stats[5].item())
    return {"loss": stats[0].item(), "train_tokens": int(stats[1].item()),
            "mismatch_abs_lp": stats[2].item() / npre, "clip_frac_0.2": stats[3].item() / ntok,
            "kl_rollout_k1": stats[4].item() / npre,
            "grad_norm": stats[6].item() / max(1.0, stats[7].item()),
            "shift_vs_base_abs_lp": stats[8].item() / stats[9].item() if stats[9].item() > 0 else None,
            "optimizer_updates": int(round(stats[7].item() / d.world))}


def _scatter(lp: torch.Tensor, b, mb: List[int], out: Dict[int, torch.Tensor]) -> None:
    off = 0
    for j, i in enumerate(mb):
        L = int(b.seq_len[j])
        out[i] = lp[off:off + L].detach()
        off += L


# ---------------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--algo", required=True, help="directory with algorithm.py exporting ALGORITHM")
    ap.add_argument("--protocol", default="configs/protocol/qwen3_8b_40960.yaml")
    ap.add_argument("--servers", required=True, help="state file written by selectrl.rollout.servers launch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-tokens", type=int, default=int(os.environ.get("SELECTRL_MICRO_TOKENS", "16384")))
    ap.add_argument("--stop-after-step", type=int, default=int(os.environ.get("SELECTRL_STOP_AFTER_STEP", "0")),
                    help="exit after this step without finishing (used by scripts/smoke.sh to test resuming)")
    args = ap.parse_args()

    from selectrl.bench.guard import load_algorithm
    from selectrl.train.ckpt import export_for_vllm, load_resume, prune_adapters, restore_weights, save_resume
    from selectrl.train.model import PolicyModel

    d = Dist()
    cfg = load_protocol(args.protocol)
    phash = protocol_hash(cfg)
    run_dir = resolve(args.run_dir)
    tr = cfg["train"]
    total = int(tr["steps"])
    algo, algo_info = load_algorithm(resolve(args.algo))
    if d.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "steps").mkdir(exist_ok=True)
        (run_dir / "adapters").mkdir(exist_ok=True)
        man_p = run_dir / "manifest.json"
        man = json.loads(man_p.read_text()) if man_p.exists() else {}
        if man and (man.get("protocol_hash") != phash or man.get("algo_sha256") != algo_info["sha256"]
                    or man.get("seed") != args.seed):
            raise SystemExit(f"{run_dir} belongs to a different protocol/algorithm/seed; use a new --run-dir")
        if not man:
            atomic_write_json(man_p, {"run_name": run_dir.name, "protocol": cfg["protocol"], "protocol_hash": phash,
                                      "algo_dir": str(args.algo), "algo_name": algo.name,
                                      "algo_sha256": algo_info["sha256"], "seed": args.seed,
                                      "world_size": d.world, "micro_tokens": args.micro_tokens,
                                      "model": cfg["model"]["hf_id"], "model_revision": cfg["model"]["revision"],
                                      "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    d.barrier()
    if (run_dir / "done").exists():
        if d.rank == 0:
            log(f"[train] {run_dir} already finished (done marker present)")
        return

    torch.manual_seed(stable_seed("init", args.seed))
    model = PolicyModel(model_path(cfg), tr["lora"], d.device)
    if d.rank == 0:
        log(f"[train] LoRA params {model.num_trainable() / 1e6:.1f}M, regex {model.target_regex}")
    betas = tuple(tr.get("adam_betas", [0.9, 0.99]))
    optimizer = torch.optim.AdamW(model.trainable, lr=float(tr["lr"]), betas=betas,
                                  eps=float(tr.get("adam_eps", 1e-8)), weight_decay=float(tr["weight_decay"]))
    from selectrl.rollout.prompts import ChatRenderer

    pad_id = ChatRenderer(model_path(cfg)).pad_id

    state = load_resume(run_dir) or {}
    start = int(state.get("step", 0)) + 1
    if state:
        restore_weights(model.peft, optimizer, run_dir, d.device)
        algo.state = state.get("algo_state", {})
        if d.rank == 0:
            log(f"[train] resumed after step {start - 1}")

    roll = Rollouts(cfg, run_dir, resolve(args.servers), args.seed, algo, state) if d.rank == 0 else None
    adapter_name: Optional[str] = None
    if d.rank == 0 and state:
        adapter_name = f"resume_{start - 1:04d}"
        roll.pool.load_adapter(adapter_name, run_dir / "resume" / "adapter_vllm")

    for step in range(start, total + 1):
        t0 = time.time()
        bpath = run_dir / "steps" / f"step_{step:04d}" / "batch.pt"
        if d.rank == 0:
            if bpath.exists():
                batch = torch.load(bpath, weights_only=False)
                roll.cursor, roll.epoch = batch["cursor"], batch["epoch"]
                roll.sigma_plus = batch["sigma_plus"]
                algo.state = batch["algo_state"]
                log(f"[train] step {step}: reusing saved rollout batch")
            else:
                batch = roll.make_batch(step, adapter_name)
                bpath.parent.mkdir(parents=True, exist_ok=True)
                tmp = bpath.with_suffix(".tmp")
                torch.save(batch, tmp)
                os.replace(tmp, bpath)
                with gzip.open(bpath.parent / "samples.jsonl.gz", "wt") as f:
                    for row in batch["log_rows"]:
                        f.write(json.dumps(row) + "\n")
            m = batch["metrics"]
            log(f"[train] step {step}/{total}: g {m['g_rate']:.3f} chk {m['chk_rate']:.3f} solved_any "
                f"{m['solved_any_rate']:.2f} kept {m['kept_seqs']} seqs, gen {m['t_generate_s']}s "
                f"exec {m['t_exec_s']}s")
        d.barrier()
        if d.rank != 0:
            batch = torch.load(bpath, weights_only=False)
            algo.state = batch["algo_state"]
        upd = train_on_batch(batch, model, optimizer, algo, d, args.micro_tokens, float(tr["max_grad_norm"]),
                             pad_id) if batch["seqs"] else {"loss": 0.0, "optimizer_updates": 0}
        d.barrier()
        mm = float(upd.get("mismatch_abs_lp", 0.0))
        limit = float(os.environ.get("SELECTRL_MAX_LP_MISMATCH", "0.08"))
        if step > 1 and mm > limit and (step <= start + 2):
            raise RuntimeError(
                f"step {step}: the inference servers' log-probs differ from the trained policy by {mm:.3f} on average "
                f"(limit {limit}); the LoRA adapter is probably not applied by vLLM. Check the adapter key layout "
                f"(selectrl.train.ckpt.export_for_vllm) and the vLLM version.")
        if d.rank == 0:
            name = f"step_{step:04d}"
            served = export_for_vllm(model.peft, run_dir / "adapters" / name, model.vllm_key_map,
                                     tr["lora"]["target_modules"])
            if step % int(tr["save_every"]) == 0 or step == total:
                (run_dir / "ckpt" / name).mkdir(parents=True, exist_ok=True)
                export_for_vllm(model.peft, run_dir / "ckpt" / name / "adapter", model.vllm_key_map,
                                tr["lora"]["target_modules"])
            if step < total:
                roll.pool.swap_adapter(name, run_dir / "adapters" / name, adapter_name)
                adapter_name = name
            save_resume(run_dir, model.peft, optimizer, {
                "step": step, "cursor": batch["cursor"], "epoch": batch["epoch"],
                "sigma_plus": batch["sigma_plus"], "algo_state": algo.state, "protocol_hash": phash},
                vllm_export=served)
            prune_adapters(run_dir / "adapters", keep=2)
            row = {"step": step, **batch["metrics"], **upd, "t_step_s": round(time.time() - t0, 1)}
            append_metrics(run_dir / "metrics.jsonl", row)
            # only the last two step batches are kept (they are large); samples.jsonl.gz is kept
            old = run_dir / "steps" / f"step_{step - 2:04d}" / "batch.pt"
            if old.exists():
                old.unlink()
        d.barrier()
        if args.stop_after_step and step >= args.stop_after_step:
            if d.rank == 0:
                log(f"[train] --stop-after-step {args.stop_after_step}: exiting for a resume test")
            return
    if d.rank == 0:
        atomic_write_json(run_dir / "done", {"steps": total, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        log(f"[train] finished {total} steps -> {run_dir}")
    d.barrier()
    if d.world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

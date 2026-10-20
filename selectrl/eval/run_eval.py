"""Evaluate a policy (base model or a LoRA adapter) on a split with the locked protocol.

    python -m selectrl.eval.run_eval --protocol configs/protocol/qwen3_8b_40960.yaml --split test \
        --servers runs/x/servers.json --adapter runs/x/ckpt/step_0100/adapter --out runs/x/eval/test_step_0100

For every problem: ``n_eval`` samples (seeded per problem, identical seeds for every policy), execution with the
frozen checker, then per-problem metrics. Writes ``<out>.jsonl`` (one row per problem, resumable: finished problems
are skipped on a rerun), ``<out>.samples.jsonl.gz`` (programs and signals, for audits) and ``<out>.summary.json``.
This module never imports algorithm code.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from selectrl.config import load_protocol, model_path, protocol_hash, resolve
from selectrl.utils import atomic_write_json, log


def _jsonl_gz(path: Path) -> List[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def problem_row(pid: str, prob: dict, recs: List[dict], choices: List[dict], k_list: List[int]) -> dict:
    from selectrl.algolib.combin import pass_at_k
    from selectrl.algolib.selector import exact_cov_at_k, exact_select_at_k

    n = len(recs)
    g = np.array([r["g"] for r in recs], dtype=np.int64)
    chk = np.array([r["chk"] for r in recs], dtype=np.int64)
    sig_map: Dict[str, int] = {}
    sig = np.array([sig_map.setdefault(r["sig"], len(sig_map)) if r["has_code"] else -1 for r in recs],
                   dtype=np.int64)
    c = int(g.sum())
    row = {
        "pid": pid, "contest_id": prob.get("contest_id"), "difficulty": prob.get("difficulty"), "n_eval": n,
        "n_correct": c, "n_chk": int(chk.sum()), "n_C": int(((g == 1) & (chk == 1)).sum()),
        "n_false_reject": int(((g == 1) & (chk == 0)).sum()),
        "n_signatures": len(sig_map),
        "has_code_rate": float(np.mean([r["has_code"] for r in recs])),
        "truncated_rate": float(np.mean([ch["finish_reason"] == "length" for ch in choices])),
        "resp_len_mean": float(np.mean([len(ch["token_ids"]) for ch in choices])),
        "pass": {}, "select": {}, "cov": {},
    }
    for k in k_list:
        if k > n:
            continue
        row["pass"][str(k)] = float(pass_at_k(n, c, k))
        row["select"][str(k)] = float(exact_select_at_k(g, chk, sig, k))
        row["cov"][str(k)] = float(exact_cov_at_k(g, chk, k))
    return row


def summarize(rows: List[dict], k_list: List[int]) -> dict:
    out = {"n_problems": len(rows)}
    for key in ("pass", "select", "cov"):
        for k in k_list:
            vals = [r[key][str(k)] for r in rows if str(k) in r[key]]
            if vals:
                out[f"{key}@{k}"] = float(np.mean(vals))
    for key in ("has_code_rate", "truncated_rate", "resp_len_mean"):
        out[key] = float(np.mean([r[key] for r in rows])) if rows else 0.0
    tot = sum(r["n_eval"] for r in rows) or 1
    out["chk_rate"] = sum(r["n_chk"] for r in rows) / tot
    out["correct_rate"] = sum(r["n_correct"] for r in rows) / tot
    out["false_reject_rate"] = sum(r["n_false_reject"] for r in rows) / tot
    return out


def run(args: argparse.Namespace) -> dict:
    from selectrl.checker.cache import ExecCache
    from selectrl.checker.chk import last_code_block as extract_program
    from selectrl.rollout.client import SampleRequest, ServerPool
    from selectrl.rollout.prompts import ChatRenderer

    cfg = load_protocol(args.protocol)
    ev, sp = cfg["eval"], cfg["sampling"]
    split_path = resolve(cfg["data"][args.split])
    if not split_path.exists():
        raise SystemExit(f"split file {split_path} not found (hidden test data is not part of agent workspaces)")
    probs = _jsonl_gz(split_path)
    if ev.get("max_problems"):
        probs = probs[: int(ev["max_problems"])]
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    k_list = [int(k) for k in ev["k_list"]]
    if args.merge:
        return merge(out, probs, k_list, cfg, args)
    prefix = str(out) if args.nshards == 1 else f"{out}.shard{args.shard}of{args.nshards}"
    rows_path = Path(prefix + ".jsonl")
    samples_path = Path(prefix + ".samples.jsonl.gz")
    if args.nshards > 1:
        probs = [p for i, p in enumerate(sorted(probs, key=lambda q: q["pid"])) if i % args.nshards == args.shard]
    done: Dict[str, dict] = {}
    if rows_path.exists():
        for l in rows_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done[r["pid"]] = r
    todo = [p for p in probs if p["pid"] not in done]
    log(f"[eval] {args.split}: {len(probs)} problems, {len(done)} already done, {len(todo)} to run")

    n_eval = int(ev["n_eval"])
    pool = ServerPool.from_state(resolve(args.servers))
    model_name: Optional[str] = None
    if args.adapter and args.adapter != "none":
        model_name = "eval_" + hashlib.sha1(str(resolve(args.adapter)).encode()).hexdigest()[:10]
        pool.load_adapter(model_name, resolve(args.adapter))
    render = ChatRenderer(model_path(cfg), enable_thinking=bool(cfg["prompt"]["enable_thinking"]))
    execc = ExecCache(resolve("cache"), cfg["checker"]["version"])
    from selectrl.train.main import load_cases

    t0 = time.time()
    chunk = max(1, int(args.chunk))
    for c0 in range(0, len(todo), chunk):
        part = todo[c0:c0 + chunk]
        reqs = [SampleRequest(list(render.solve_prompt_ids(p["question"])), n_eval, int(sp["max_new_tokens"]),
                              float(sp["temperature"]), float(sp["top_p"]), int(sp["top_k"]),
                              _seed("eval", ev["seed"], p["pid"]), model_name, p["pid"]) for p in part]
        outs = pool.sample_many(reqs)
        items = []
        for p, choices in zip(part, outs):
            cases = load_cases(resolve("cache"), p["pid"])
            items += [(p, cases, extract_program(ch["text"])) for ch in choices]
        recs = execc.evaluate_many(items)
        new_rows = []
        with gzip.open(samples_path, "at") as fz:
            for i, (p, choices) in enumerate(zip(part, outs)):
                rr = recs[i * n_eval:(i + 1) * n_eval]
                new_rows.append(problem_row(p["pid"], p, rr, choices, k_list))
                for j, (ch, r) in enumerate(zip(choices, rr)):
                    fz.write(json.dumps({"pid": p["pid"], "j": j, "finish": ch["finish_reason"],
                                         "len": len(ch["token_ids"]), "g": r["g"], "chk": r["chk"],
                                         "sig": r["sig"], "code": items[i * n_eval + j][2]}) + "\n")
        with open(rows_path, "a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
                done[r["pid"]] = r
        log(f"[eval] {len(done)}/{len(probs)} problems ({time.time() - t0:.0f}s)")
    if model_name:
        pool.unload_adapter(model_name)
    if args.nshards > 1:
        log(f"[eval] shard {args.shard}/{args.nshards} done ({len(done)} problems); merge with --merge")
        return {}
    rows = [done[p["pid"]] for p in probs if p["pid"] in done]
    return write_summary(out, rows, k_list, cfg, args)


def write_summary(out: Path, rows: List[dict], k_list: List[int], cfg: dict, args) -> dict:
    ev = cfg["eval"]
    n_eval = int(ev["n_eval"])
    summ = summarize(rows, k_list)
    summ.update({"split": args.split, "adapter": str(args.adapter), "protocol": cfg["protocol"],
                 "protocol_hash": protocol_hash(cfg), "n_eval": n_eval, "primary_metric": ev["primary_metric"],
                 "primary": summ.get(ev["primary_metric"])})
    atomic_write_json(out.with_suffix(".summary.json"), summ)
    log(f"[eval] {args.split} {ev['primary_metric']} = {summ.get(ev['primary_metric'])}")
    return summ


def merge(out: Path, probs: List[dict], k_list: List[int], cfg: dict, args) -> dict:
    """Combine the shard files of a sharded evaluation into ``<out>.jsonl`` / ``.samples.jsonl.gz`` / summary."""
    rows: Dict[str, dict] = {}
    for f in sorted(out.parent.glob(out.name + ".shard*of*.jsonl")) + [Path(f"{out}.jsonl")]:
        if f.exists():
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    rows[r["pid"]] = r
    missing = [p["pid"] for p in probs if p["pid"] not in rows]
    if missing:
        raise SystemExit(f"merge: {len(missing)} problems not evaluated yet (e.g. {missing[:3]})")
    ordered = [rows[p["pid"]] for p in probs]
    tmp = Path(f"{out}.jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in ordered))
    os.replace(tmp, Path(f"{out}.jsonl"))
    shards = sorted(out.parent.glob(out.name + ".shard*of*.samples.jsonl.gz"))
    if shards:
        with gzip.open(Path(f"{out}.samples.jsonl.gz"), "wt") as fo:
            for f in shards:
                with gzip.open(f, "rt") as fi:
                    for l in fi:
                        fo.write(l)
    for f in sorted(out.parent.glob(out.name + ".shard*of*.*")):
        f.unlink()
    return write_summary(out, ordered, k_list, cfg, args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="configs/protocol/qwen3_8b_40960.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--servers", default=None, help="state file of selectrl.rollout.servers launch (not for --merge)")
    ap.add_argument("--adapter", default="none", help="LoRA adapter directory, or 'none' for the base model")
    ap.add_argument("--out", required=True, help="output prefix, e.g. runs/x/eval/test_step_0100")
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SELECTRL_EVAL_CHUNK", "64")))
    ap.add_argument("--shard", type=int, default=0, help="evaluate problems i::nshards only (one job per GPU)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="combine finished shards and write the summary")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

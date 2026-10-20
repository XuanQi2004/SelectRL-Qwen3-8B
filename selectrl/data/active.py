"""Select the active training pool: candidate training problems the base model solves sometimes but not always.

    python -m selectrl.data.active --servers runs/_active/servers.json \
        --candidates data/problems/train_candidates.jsonl.gz --max-candidates 4000 --n 8 \
        --out data/problems/train.jsonl.gz

``n`` base-model samples per candidate (protocol sampling parameters, thinking off) are labeled with the hidden
tests only (no checker needed); a problem is kept when 1 <= #correct <= n - 1. Per-problem base pass counts are
written to ``data/active/base_pass.jsonl`` (resumable) and kept for reference. Maintainers only: the shipped data
already contains the selected pool.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from selectrl.config import load_protocol, model_path, resolve
from selectrl.utils import log


def _rows(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _seed(*p) -> int:
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="configs/protocol/qwen3_8b_40960.yaml")
    ap.add_argument("--servers", default=None)
    ap.add_argument("--candidates", default="data/problems/train_candidates.jsonl.gz")
    ap.add_argument("--max-candidates", type=int, default=4000)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="data/problems/train.jsonl.gz")
    ap.add_argument("--stats", default="data/active/base_pass.jsonl")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0, help="process candidates i::nshards (one job per GPU)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--select-only", action="store_true",
                    help="no sampling: merge every base_pass*.jsonl next to --stats and write the pool")
    a = ap.parse_args()

    from selectrl.checker.chk import label_program_on_hidden, last_code_block
    from selectrl.rollout.client import SampleRequest, ServerPool
    from selectrl.rollout.prompts import ChatRenderer
    from selectrl.sandbox.pool import pmap

    cfg = load_protocol(a.protocol)
    sp = cfg["sampling"]
    cands = _rows(resolve(a.candidates))
    cands.sort(key=lambda p: hashlib.sha256(f"selectrl-active|{p['pid']}".encode()).hexdigest())
    cands = cands[: a.max_candidates]
    stats_p = resolve(a.stats)
    stats_p.parent.mkdir(parents=True, exist_ok=True)
    if a.nshards > 1 and not a.select_only:
        stats_p = stats_p.with_name(f"{stats_p.stem}.shard{a.shard}of{a.nshards}.jsonl")
    done = {}
    for f in ([stats_p] if not a.select_only else sorted(stats_p.parent.glob(stats_p.stem + "*.jsonl"))):
        if f.exists():
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    done[r["pid"]] = r
    if a.select_only:
        missing = [p["pid"] for p in cands if p["pid"] not in done]
        if missing:
            raise SystemExit(f"{len(missing)} candidates have no base pass statistics yet (e.g. {missing[:3]})")
        todo = []
    else:
        todo = [p for i, p in enumerate(cands) if i % a.nshards == a.shard and p["pid"] not in done]
    log(f"[active] {len(cands)} candidates, {len(done)} done, {len(todo)} to sample (n={a.n})")
    if todo:
        pool = ServerPool.from_state(resolve(a.servers))
        render = ChatRenderer(model_path(cfg), enable_thinking=bool(cfg["prompt"]["enable_thinking"]))
    t0 = time.time()
    for c0 in range(0, len(todo), a.chunk):
        part = todo[c0:c0 + a.chunk]
        reqs = [SampleRequest(list(render.solve_prompt_ids(p["question"])), a.n, int(sp["max_new_tokens"]),
                              float(sp["temperature"]), float(sp["top_p"]), int(sp["top_k"]),
                              _seed("active", p["pid"]), None, p["pid"]) for p in part]
        outs = pool.sample_many(reqs)
        jobs = [(p, last_code_block(ch["text"])) for p, chs in zip(part, outs) for ch in chs]
        labels = pmap(lambda x: label_program_on_hidden(x[1], x[0]), jobs)
        with open(stats_p, "a") as f:
            for i, (p, chs) in enumerate(zip(part, outs)):
                gs = [int(bool(labels[i * a.n + j].get("all"))) for j in range(a.n)]
                r = {"pid": p["pid"], "n": a.n, "c": sum(gs), "trunc": sum(ch["finish_reason"] == "length" for ch in chs),
                     "len_mean": sum(len(ch["token_ids"]) for ch in chs) / a.n}
                done[p["pid"]] = r
                f.write(json.dumps(r) + "\n")
        log(f"[active] {len(done)}/{len(cands)} ({time.time() - t0:.0f}s)")
    if a.nshards > 1 and not a.select_only:
        log(f"[active] shard {a.shard}/{a.nshards} finished; merge with --select-only")
        return
    keep = [p for p in cands if 1 <= done[p["pid"]]["c"] <= a.n - 1]
    out = resolve(a.out)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for p in sorted(keep, key=lambda x: x["pid"]):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    hist = {}
    for r in done.values():
        hist[r["c"]] = hist.get(r["c"], 0) + 1
    log(f"[active] kept {len(keep)} / {len(cands)}; histogram of #correct: {dict(sorted(hist.items()))} -> {out}")
    from selectrl.utils import atomic_write_json

    atomic_write_json(resolve(a.stats).with_name("base_pass_summary.json"), {
        "candidates": len(cands), "max_candidates": a.max_candidates, "samples_per_problem": a.n,
        "kept": len(keep), "rule": "1 <= correct <= n-1", "histogram_correct": {str(k): v for k, v in sorted(hist.items())},
        "sampling": {k: sp[k] for k in ("temperature", "top_p", "top_k", "max_new_tokens")},
        "base_pass_at_1_candidates": sum(r["c"] for r in done.values()) / max(1, a.n * len(done)),
    })


if __name__ == "__main__":
    main()

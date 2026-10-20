"""Download ``deepmind/code_contests``, convert, filter and split it into the benchmark's problem files.

Ported from SSPO v1 (data_pipeline.py convert_code_contests_row / prepare_datasets).

Outputs under ``--out`` (default ``data``):
  problems/{train_candidates,val,test}.jsonl.gz   problem records (schema: docs/ARCHITECTURE.md section 3)
  splits/{train_candidates,val,test}.txt          sorted pids
  filter_stats.json                               conversion, filter and split statistics

Usage: python -m selectrl.data.codecontests --out data [--max-train N] [--streaming]
The Hugging Face token is read from ``HF_TOKEN`` when set; ``HF_ENDPOINT`` is honoured by ``huggingface_hub``.
Rows are converted and filtered one at a time and staged on disk, so memory stays small even though the
generated tests of the full dataset are several GB.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from selectrl.utils import atomic_open, atomic_write_json, atomic_write_jsonl_gz, iter_jsonl, log, sha1_16

from .filters import FilterStats
from .splits import N_TEST_HELDOUT, N_VAL_HELDOUT, SPLITS, assign_splits, check_contest_disjoint, write_split_lists

DATASET = "deepmind/code_contests"
COLUMNS = ["name", "description", "public_tests", "private_tests", "generated_tests", "source", "difficulty",
           "solutions", "cf_contest_id", "cf_rating"]
ORIG_SPLITS = ("test", "valid", "train")  # processing order: evaluation problems win pid collisions
PLATFORMS = {0: "unknown", 1: "codechef", 2: "codeforces", 3: "hackerearth", 4: "codejam", 5: "atcoder", 6: "aizu"}
CODEFORCES = 2
PYTHON3 = 3
MAX_HIDDEN = 40
MAX_REFS = 3
MAX_TEST_BYTES = 4 << 20
TIME_LIMIT_S = 6.0

_ID_TOKEN = re.compile(r"^(\d+_[A-Z]\d*|p\d{5})$")  # Codeforces "1549_C", AtCoder / Aizu "p02543"
_CF_NAME = re.compile(r"^(\d+)_[A-Z]")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def make_pid(name: str) -> str:
    """``cc-<problem id>`` for id-like names (``1549_C. Web of Lies`` -> ``cc-1549_C``), else slug + short hash."""
    token = name.split()[0].rstrip(".") if name.split() else ""
    if _ID_TOKEN.match(token):
        return f"cc-{token}"
    return f"cc-{_slug(name)[:48]}-{sha1_16(name)[:6]}"


def make_contest_id(row: dict[str, Any]) -> str:
    """``cf_<id>`` from ``cf_contest_id`` (or a Codeforces name), else ``<platform>_<name prefix>``.

    The name prefix drops a leading ``pNNNNN`` id and keeps the part before ``" - "`` (AtCoder contest titles).
    """
    name = str(row.get("name") or "")
    cf = int(row.get("cf_contest_id") or 0)
    m = _CF_NAME.match(name)
    if cf > 0:
        return f"cf_{cf}"
    if int(row.get("source") or 0) == CODEFORCES and m:
        return f"cf_{m.group(1)}"
    prefix = re.sub(r"^p\d{5}\s+", "", name).split(" - ")[0]
    return f"{PLATFORMS.get(int(row.get('source') or 0), 'unknown')}_{_slug(prefix).lower()[:64]}"


def _tests(block: dict[str, Any] | None) -> tuple[list[list[str]], int]:
    """``[[input, output]]`` of a test block without tests whose input or output exceeds 4 MiB; also the skip count."""
    block = block or {}
    tests, skipped = [], 0
    for i, o in zip(block.get("input") or [], block.get("output") or []):
        i, o = str(i), str(o)
        if len(i.encode("utf-8")) > MAX_TEST_BYTES or len(o.encode("utf-8")) > MAX_TEST_BYTES:
            skipped += 1
        else:
            tests.append([i, o])
    return tests, skipped


def convert_row(row: dict[str, Any], orig_split: str) -> tuple[dict[str, Any], int]:
    """Problem record of one dataset row and the number of tests skipped for size."""
    name = str(row.get("name") or "")
    public, s1 = _tests(row.get("public_tests"))
    private, s2 = _tests(row.get("private_tests"))
    generated, s3 = _tests(row.get("generated_tests"))
    sols = row.get("solutions") or {}
    refs = [str(c) for lang, c in zip(sols.get("language") or [], sols.get("solution") or [])
            if int(lang) == PYTHON3 and c][:MAX_REFS]
    rating = int(row.get("cf_rating") or 0)
    level = int(row.get("difficulty") or 0)
    prob = {
        "pid": make_pid(name),
        "source": "code_contests",
        "platform": PLATFORMS.get(int(row.get("source") or 0), "unknown"),
        "orig_split": orig_split,
        "name": name,
        "contest_id": make_contest_id(row),
        "difficulty": str(rating) if rating > 0 else (str(level) if level > 0 else "unknown"),
        "time_limit_s": TIME_LIMIT_S,
        "question": str(row.get("description") or ""),
        "public": public,
        "hidden": (private + generated)[:MAX_HIDDEN],
        "py_refs": refs,
    }
    return prob, s1 + s2 + s3


def load_rows(orig_split: str, streaming: bool = False, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    """Rows of one CodeContests split (network access through ``datasets``)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=orig_split, streaming=streaming, token=os.environ.get("HF_TOKEN") or None)
    rows: Iterable[dict[str, Any]] = ds.select_columns(COLUMNS)
    return iter(itertools.islice(rows, max_rows) if max_rows is not None else rows)


def _stage(rows: Iterable[dict[str, Any]], orig: str, seen: set[str], meta: list[dict[str, str]],
           fstats: FilterStats, conv: Counter) -> Iterator[dict[str, Any]]:
    """Converted rows of one original split that pass the filter (statistics are updated in place)."""
    for row in rows:
        prob, n_skipped = convert_row(row, orig)
        conv["rows"] += 1
        conv["tests_skipped_oversize"] += n_skipped
        conv["problems_without_py_refs"] += not prob["py_refs"]
        if prob["pid"] in seen:
            fstats.add(prob, "duplicate_pid")
            continue
        seen.add(prob["pid"])
        if fstats.add(prob):
            meta.append({"pid": prob["pid"], "contest_id": prob["contest_id"], "orig_split": orig})
            yield prob


def prepare(out_dir: str | Path, rows_by_split: dict[str, Iterable[dict[str, Any]]],
            n_test_heldout: int = N_TEST_HELDOUT, n_val_heldout: int = N_VAL_HELDOUT) -> dict[str, Any]:
    """Convert, filter and split the rows of each original split (``train`` / ``valid`` / ``test``); write outputs."""
    out = Path(out_dir)
    staging = out / "problems" / ".staging"
    seen: set[str] = set()
    meta: list[dict[str, str]] = []
    stats: dict[str, Any] = {"per_orig_split": {}, "conversion": {}}
    for orig in ORIG_SPLITS:
        fstats, conv = FilterStats(), Counter()
        atomic_write_jsonl_gz(staging / f"{orig}.jsonl.gz",
                              _stage(rows_by_split.get(orig, ()), orig, seen, meta, fstats, conv))
        stats["per_orig_split"][orig] = fstats.as_dict()
        stats["conversion"][orig] = dict(conv)
        log(f"[data] {orig}: {conv['rows']} rows, kept {stats['per_orig_split'][orig]['kept']}")

    assignment, split_info = assign_splits(meta, n_test_heldout, n_val_heldout)
    check_contest_disjoint(meta, assignment)
    with contextlib.ExitStack() as stack:
        files = {s: stack.enter_context(atomic_open(out / "problems" / f"{s}.jsonl.gz", gz=True)) for s in SPLITS}
        for orig in ORIG_SPLITS:
            for prob in iter_jsonl(staging / f"{orig}.jsonl.gz"):
                files[assignment[prob["pid"]]].write(json.dumps(prob, ensure_ascii=False).encode("utf-8") + b"\n")
    shutil.rmtree(staging, ignore_errors=True)
    write_split_lists(out, assignment)
    stats["splits"] = split_info
    atomic_write_json(out / "filter_stats.json", stats)
    log(f"[data] split counts {split_info['counts']}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Download, convert, filter and split deepmind/code_contests.")
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--max-train", type=int, default=None, help="read at most N train rows (smoke runs)")
    ap.add_argument("--streaming", action="store_true", help="stream rows instead of downloading the dataset")
    ap.add_argument("--n-test-heldout", type=int, default=N_TEST_HELDOUT)
    ap.add_argument("--n-val-heldout", type=int, default=N_VAL_HELDOUT)
    args = ap.parse_args()
    rows = {s: load_rows(s, args.streaming, args.max_train if s == "train" else None) for s in ORIG_SPLITS}
    prepare(args.out, rows, args.n_test_heldout, args.n_val_heldout)


if __name__ == "__main__":
    main()

"""Contest-disjoint train / val / test splits.

val  = filtered CodeContests ``valid`` + about ``N_VAL_HELDOUT`` problems of whole held-out ``train`` contests,
test = filtered CodeContests ``test``  + about ``N_TEST_HELDOUT`` problems of whole held-out ``train`` contests,
train_candidates = every other filtered ``train`` problem.

Held-out contests are taken in the order of ``sha256(f"{SALT}:{contest_id}")`` (test first, then val) until the
target number of problems is reached, so the choice is deterministic and independent of the row order. A contest
never spans two splits: a ``train`` or ``valid`` problem whose contest occurs in a later-priority split
(test > val) moves there with its contest.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from selectrl.utils import atomic_write_text

SALT = "selectrl-v1"
N_TEST_HELDOUT = 300
N_VAL_HELDOUT = 180
SPLITS = ("train_candidates", "val", "test")
_ORIG_TO_SPLIT = {"test": "test", "valid": "val", "train": "train_candidates"}
_PRIORITY = {"test": 2, "val": 1, "train_candidates": 0}


def contest_rank(contest_id: str) -> str:
    """Deterministic ordering key of a contest for hold-out selection."""
    return hashlib.sha256(f"{SALT}:{contest_id}".encode("utf-8")).hexdigest()


def assign_splits(problems: Iterable[dict[str, Any]], n_test_heldout: int = N_TEST_HELDOUT,
                  n_val_heldout: int = N_VAL_HELDOUT) -> tuple[dict[str, str], dict[str, Any]]:
    """Map ``pid -> split`` for problems given as ``{"pid", "contest_id", "orig_split"}``, plus split statistics."""
    probs = list(problems)
    contest_split: dict[str, str] = {}
    for p in probs:  # contests of the official evaluation splits (test wins over valid)
        s = _ORIG_TO_SPLIT[p["orig_split"]]
        if s != "train_candidates" and _PRIORITY[s] > _PRIORITY.get(contest_split.get(p["contest_id"], ""), -1):
            contest_split[p["contest_id"]] = s

    train_contests: dict[str, int] = defaultdict(int)
    for p in probs:
        if p["orig_split"] == "train" and p["contest_id"] not in contest_split:
            train_contests[p["contest_id"]] += 1
    added = {"test": 0, "val": 0}
    heldout: dict[str, list[str]] = {"test": [], "val": []}
    targets = {"test": n_test_heldout, "val": n_val_heldout}
    for cid in sorted(train_contests, key=contest_rank):
        split = next((s for s in ("test", "val") if added[s] < targets[s]), None)
        if split is None:
            break
        contest_split[cid] = split
        added[split] += train_contests[cid]
        heldout[split].append(cid)

    heldout_ids = {cid for cids in heldout.values() for cid in cids}
    assignment: dict[str, str] = {}
    moved: dict[str, int] = defaultdict(int)
    for p in probs:
        s = contest_split.get(p["contest_id"], "train_candidates")
        assignment[p["pid"]] = s
        if p["orig_split"] != "train" and s != _ORIG_TO_SPLIT[p["orig_split"]]:
            moved[f"{p['orig_split']}->{s}"] += 1
        elif p["orig_split"] == "train" and s != "train_candidates" and p["contest_id"] not in heldout_ids:
            moved[f"train->{s}"] += 1
    counts = {s: sum(v == s for v in assignment.values()) for s in SPLITS}
    info = {
        "salt": SALT,
        "counts": counts,
        "heldout_test_contests": sorted(heldout["test"]),
        "heldout_val_contests": sorted(heldout["val"]),
        "n_heldout_test_problems": added["test"],
        "n_heldout_val_problems": added["val"],
        "moved_with_contest": dict(sorted(moved.items())),
    }
    return assignment, info


def check_contest_disjoint(problems: Iterable[dict[str, Any]], assignment: dict[str, str]) -> None:
    """Raise if any contest spans two splits."""
    seen: dict[str, str] = {}
    for p in problems:
        s = assignment[p["pid"]]
        if seen.setdefault(p["contest_id"], s) != s:
            raise AssertionError(f"contest {p['contest_id']} is in both {seen[p['contest_id']]} and {s}")


def write_split_lists(out_dir: str | Path, assignment: dict[str, str]) -> dict[str, Path]:
    """Write ``<out_dir>/splits/<split>.txt`` (sorted pids, one per line) for every split."""
    paths = {}
    for s in SPLITS:
        pids = sorted(pid for pid, v in assignment.items() if v == s)
        paths[s] = Path(out_dir) / "splits" / f"{s}.txt"
        atomic_write_text(paths[s], "".join(f"{pid}\n" for pid in pids))
    return paths

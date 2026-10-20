"""Persistent execution cache keyed by (pid, code_hash) for one checker version.

Ported from SSPO v1 (checker.py ExecutionCache).

Layout: ``<root>/exec/<checker_version>/<pid>.json`` = ``{code_hash: execution record}`` (``root`` is the cache
directory, e.g. ``cache``). Writes merge with the file's current content under an exclusive ``flock`` on a sidecar
lock file and replace it atomically, so several threads or processes can share one cache without losing records.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import threading
from pathlib import Path
from typing import Any, Iterator, Sequence

from selectrl.sandbox import pmap
from selectrl.utils import atomic_write_json, read_json

from .build import CHECKER_VERSION
from .chk import NOCODE_KEY, code_hash, evaluate_program, nocode_record

Item = tuple[dict[str, Any], dict[str, Any], str | None]


class ExecCache:
    """Thread-safe execution cache with hit / miss counters.

    ``hits`` counts items answered without execution (cached, duplicate within a call, or without code);
    ``misses`` counts distinct (pid, code_hash) pairs that had to be executed.
    """

    def __init__(self, root: str | Path, checker_version: str) -> None:
        if checker_version != CHECKER_VERSION:
            raise ValueError(f"cache version {checker_version!r} differs from the checker's {CHECKER_VERSION!r}")
        self.dir = Path(root) / "exec" / checker_version
        self.dir.mkdir(parents=True, exist_ok=True)
        self.checker_version = checker_version
        self._mem: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of items served without execution."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset_counters(self) -> None:
        """Zero the hit / miss counters."""
        with self._lock:
            self.hits = self.misses = 0

    def _path(self, pid: str) -> Path:
        return self.dir / f"{pid}.json"

    @contextlib.contextmanager
    def _file_lock(self, pid: str) -> Iterator[None]:
        with open(self.dir / f".{pid}.lock", "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _read_file(self, pid: str) -> dict[str, dict[str, Any]]:
        p = self._path(pid)
        if not p.exists():
            return {}
        try:
            return read_json(p)
        except (OSError, json.JSONDecodeError):
            return {}

    def _load(self, pid: str) -> dict[str, dict[str, Any]]:
        """In-memory map of a problem (loaded from disk on first use); caller holds ``self._lock``."""
        if pid not in self._mem:
            self._mem[pid] = self._read_file(pid)
        return self._mem[pid]

    def get(self, pid: str, key: str) -> dict[str, Any] | None:
        """Cached record of a program, if any."""
        with self._lock:
            return self._load(pid).get(key)

    def _persist(self, pid: str, new: dict[str, dict[str, Any]]) -> None:
        with self._file_lock(pid):
            merged = self._read_file(pid)
            merged.update(new)
            atomic_write_json(self._path(pid), merged, indent=None)
        with self._lock:
            self._load(pid).update(merged)

    def evaluate_many(self, items: Sequence[Item]) -> list[dict[str, Any]]:
        """Execution records of ``(problem, cases, code)`` items in order; misses run in parallel and are persisted."""
        keys = [code_hash(code) for _, _, code in items]
        todo: dict[tuple[str, str], Item] = {}
        with self._lock:
            for item, key in zip(items, keys):
                pid = item[0]["pid"]
                if key != NOCODE_KEY and key not in self._load(pid) and (pid, key) not in todo:
                    todo[(pid, key)] = item

        results = pmap(lambda it: evaluate_program(*it), list(todo.values()))
        by_pid: dict[str, dict[str, dict[str, Any]]] = {}
        for (pid, key), rec in zip(todo, results):
            by_pid.setdefault(pid, {})[key] = rec
        for pid, recs in by_pid.items():
            self._persist(pid, recs)

        with self._lock:
            self.misses += len(todo)
            self.hits += len(items) - len(todo)
            return [nocode_record() if key == NOCODE_KEY else dict(self._load(item[0]["pid"])[key])
                    for item, key in zip(items, keys)]

    def evaluate(self, problem: dict[str, Any], cases: dict[str, Any], code: str | None) -> dict[str, Any]:
        """Execution record of one program (cached)."""
        return self.evaluate_many([(problem, cases, code)])[0]

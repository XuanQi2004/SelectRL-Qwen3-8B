"""Pool of persistent sandbox workers (``exec_worker.py``) and a thread map that keeps them busy.

Ported from SSPO v1 (ec_common.py run_prog / pmap / worker pool).

Configuration (environment):
  SELECTRL_EXEC_WORKERS   number of worker processes (default ``max(4, min(cpu_count - 4, 256))``)
  SELECTRL_EXEC_MEM_MB    address-space limit of every run in MiB (default 2048)
  SELECTRL_UNSHARE_NET    ``auto`` (default: wrap workers in ``unshare -rn`` when ``unshare -rn true`` works),
                          ``1`` (always; fails loudly if unavailable) or ``0`` (never)
Workers are started lazily, receive an environment with every BLAS / OpenMP thread pool pinned to one thread,
and are restarted transparently if one dies or stops answering.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from .exec_worker import THREAD_ENV_VARS

T = TypeVar("T")
R = TypeVar("R")
Result = tuple[str, str, float, float]

WORKER_PY = Path(__file__).resolve().with_name("exec_worker.py")
DEFAULT_OUT_LIMIT = 1 << 20
_JOB_SLACK_S = 60.0


def default_workers() -> int:
    """Pool size: SELECTRL_EXEC_WORKERS, else ``max(4, min(usable_cpus - 4, 256))``.

    Usable CPUs respect the process's CPU affinity (container / scheduler limits), not just the host's core count.
    """
    env = os.environ.get("SELECTRL_EXEC_WORKERS")
    if env:
        return max(1, int(env))
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        ncpu = os.cpu_count() or 8
    return max(4, min(ncpu - 4, 256))


@lru_cache(maxsize=1)
def _unshare_available() -> bool:
    if not sys.platform.startswith("linux") or not shutil.which("unshare"):
        return False
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def worker_command() -> list[str]:
    """Command line of one worker, optionally inside a fresh user + network namespace."""
    cmd = [sys.executable, str(WORKER_PY)]
    mode = os.environ.get("SELECTRL_UNSHARE_NET", "auto").strip().lower()
    if mode == "1" or (mode == "auto" and _unshare_available()):
        cmd = ["unshare", "-rn", *cmd]
    return cmd


def worker_env() -> dict[str, str]:
    """Environment of a worker: the current one with single-threaded numeric libraries and a fixed hash seed."""
    env = dict(os.environ)
    for var in THREAD_ENV_VARS:
        env[var] = "1"
    env["PYTHONHASHSEED"] = "0"  # deterministic set / dict-of-str iteration order inside programs
    return env


class _Worker:
    """One persistent worker process speaking the JSON-lines protocol of ``exec_worker``."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            worker_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=worker_env(), close_fds=True
        )
        self._buf = b""

    def request(self, job: dict, deadline_s: float) -> list[list]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(job).encode("utf-8") + b"\n")
        self.proc.stdin.flush()
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            ready, _, _ = select.select([fd], [], [], deadline_s)
            if not ready:
                raise TimeoutError(f"no answer within {deadline_s:.0f} s")
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                raise RuntimeError("worker exited")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)["res"]

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass


class ExecPool:
    """Thread-safe pool of at most ``n_workers`` lazily started sandbox workers."""

    def __init__(self, n_workers: int | None = None) -> None:
        self.n_workers = n_workers or default_workers()
        self._slots = threading.Semaphore(self.n_workers)
        self._idle: list[_Worker] = []
        self._all: list[_Worker] = []
        self._lock = threading.Lock()

    def _acquire(self) -> _Worker:
        self._slots.acquire()
        with self._lock:
            if self._idle:
                return self._idle.pop()
        try:
            w = _Worker()
        except BaseException:
            self._slots.release()
            raise
        with self._lock:
            self._all.append(w)
        return w

    def _release(self, w: _Worker, healthy: bool) -> None:
        with self._lock:
            if healthy:
                self._idle.append(w)
            else:
                self._all.remove(w)
        if not healthy:
            w.kill()
        self._slots.release()

    def run(self, job: dict) -> list[list]:
        """Send ``job`` to an idle worker; a dead or stuck worker is replaced and the job retried once."""
        n = len(job["inputs"])
        deadline = n * (3.0 * float(job["timeout"]) + 3.0) + _JOB_SLACK_S
        err: BaseException | None = None
        for _ in range(2):
            w = self._acquire()
            try:
                res = w.request(job, deadline)
            except (OSError, ValueError, RuntimeError, TimeoutError) as e:
                self._release(w, healthy=False)
                err = e
                continue
            self._release(w, healthy=True)
            return res
        return [["error", "worker failure " + repr(err)[:200], 0.0, 0.0] for _ in range(n)]

    def close(self) -> None:
        """Terminate every worker."""
        with self._lock:
            workers, self._all, self._idle = self._all, [], []
        for w in workers:
            w.kill()


_POOL: ExecPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> ExecPool:
    """The process-wide pool (created on first use)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ExecPool()
        return _POOL


def shutdown_pool() -> None:
    """Terminate the process-wide pool's workers (a later call to ``run_prog`` starts a new pool)."""
    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        pool.close()


def run_prog(
    code: str | None,
    inputs: Sequence[str],
    timeout_s: float,
    expected: Sequence[str | None] | None = None,
    early_stop: bool = False,
    keep_out: bool = False,
    out_limit: int = DEFAULT_OUT_LIMIT,
    key_only: bool = False,
) -> list[Result]:
    """Run ``code`` on every input in one sandboxed worker; returns ``[(status, out_or_detail, wall_s, cpu_s)]``.

    Status is ok / pass / wrong / error / timeout / compile_error, or nocode for every input when ``code`` is
    empty. See ``exec_worker`` for the meaning of ``expected``, ``early_stop``, ``keep_out``, ``key_only`` and
    ``out_limit`` (bytes of stdout; more is an error).
    """
    if not inputs:
        return []
    if code is None or not str(code).strip():
        return [("nocode", "", 0.0, 0.0) for _ in inputs]
    job: dict = {
        "code": code,
        "inputs": list(inputs),
        "timeout": float(timeout_s),
        "early_stop": bool(early_stop),
        "keep_out": bool(keep_out),
        "key_only": bool(key_only),
        "out_limit": int(out_limit),
    }
    if expected is not None:
        job["expected"] = list(expected)
    return [(str(r[0]), str(r[1]), float(r[2]), float(r[3])) for r in get_pool().run(job)]


def pmap(fn: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> list[R]:
    """``[fn(x) for x in items]`` evaluated by ``workers`` threads (default: the pool size), order preserved."""
    items_list = list(items)
    if not items_list:
        return []
    n = min(len(items_list), workers or get_pool().n_workers)
    if n == 1:
        return [fn(x) for x in items_list]
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(fn, items_list))

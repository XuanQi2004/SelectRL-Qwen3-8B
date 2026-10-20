"""Persistent sandboxed execution worker: reads JSON jobs (one per line) from stdin, answers one JSON line per job.

Ported from SSPO v1 (exec_worker.py).

Started by ``selectrl.sandbox.pool`` as a plain script. Before anything heavy is imported it pins every BLAS /
OpenMP / Rayon thread pool to one thread, then preloads common standard-library modules and NumPy (if importable)
so that the forked children share them copy-on-write and ``import numpy`` in a program is instant.

Job:    {"code": str, "inputs": [str, ...], "timeout": float, "expected": [str | null, ...] (optional),
         "early_stop": bool, "keep_out": bool, "key_only": bool, "out_limit": int}
Result: {"res": [[status, out_or_detail, wall_s, cpu_s], ...]}

Status: ok / pass / wrong (with "expected") / error / timeout / compile_error. With "early_stop" execution stops
after the first input whose status is not ok / pass (the result list is then shorter than the inputs).
Second field: the output for ok / pass when keep_out (else ""), the first 300 characters of the output for wrong
(full output when keep_out), the stderr tail for error, the compiler message for compile_error, "" for timeout.
With key_only it is ``out_key(output)`` for ok / pass / wrong instead.

Sandbox of every run (one forked child per input):
  new session / process group, fresh temporary working directory (also TMPDIR), Python-level socket creation
  disabled, recursion limit 10,000, RLIMIT_AS (SELECTRL_EXEC_MEM_MB, default 2048 MiB; not enforceable on macOS),
  RLIMIT_FSIZE 64 MiB, RLIMIT_CPU slightly above the time limit, output cap ``out_limit`` bytes (larger -> error).
  The verdict is based on CPU time (user + system, from wait4): CPU time above the limit or SIGXCPU -> timeout.
  Wall time is only a hang guard: a run exceeding 3 x limit + 2 s (e.g. sleeping or blocked) is killed -> timeout.
  The whole process group is killed after every run, so stray grandchildren do not survive.
"""
from __future__ import annotations

import builtins
import importlib
import io
import json
import math
import os
import resource
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
import traceback

if __package__:
    from .compare import out_key, same_output
else:  # executed as a script by the pool: this file's directory is sys.path[0]
    from compare import out_key, same_output

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
)
PRELOAD_MODULES = (
    "array", "bisect", "collections", "copy", "decimal", "fractions", "functools", "heapq", "itertools",
    "math", "operator", "random", "re", "string", "typing", "numpy",
)
FSIZE_LIMIT = 64 << 20
DEFAULT_MEM_MB = 2048
WRONG_HEAD = 300


def _blocked_socket(*_args, **_kwargs):
    raise PermissionError("network access is disabled in the selectrl sandbox")


def _disable_network() -> None:
    """Make Python-level socket creation fail inside the child."""
    socket.socket = _blocked_socket  # type: ignore[misc,assignment]
    socket.create_connection = _blocked_socket  # type: ignore[assignment]
    socket.socketpair = _blocked_socket  # type: ignore[assignment]


def _set_limits(cpu_soft: int, mem_bytes: int) -> None:
    for lim, val in (
        (resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1)),
        (resource.RLIMIT_FSIZE, (FSIZE_LIMIT, FSIZE_LIMIT)),
        (resource.RLIMIT_AS, (mem_bytes, mem_bytes)),
        (resource.RLIMIT_CORE, (0, 0)),
    ):
        try:
            resource.setrlimit(lim, val)
        except (ValueError, OSError):  # RLIMIT_AS cannot be lowered on macOS
            pass


def _child(code_obj, fds: tuple[int, int, int], work_dir: str, cpu_soft: int, mem_bytes: int) -> None:
    """Body of the forked child; never returns."""
    rc = 1
    try:
        os.setsid()
        os.chdir(work_dir)
        os.environ["TMPDIR"] = work_dir
        tempfile.tempdir = work_dir
        _disable_network()
        _set_limits(cpu_soft, mem_bytes)
        for src, dst in zip(fds, (0, 1, 2)):
            os.dup2(src, dst)
        sys.stdin = io.TextIOWrapper(io.BufferedReader(io.FileIO(0, "r", closefd=False)), encoding="utf-8")
        sys.stdout = io.TextIOWrapper(io.BufferedWriter(io.FileIO(1, "w", closefd=False)), encoding="utf-8")
        sys.stderr = io.TextIOWrapper(io.BufferedWriter(io.FileIO(2, "w", closefd=False)), encoding="utf-8")
        sys.setrecursionlimit(10000)
        g = {"__name__": "__main__", "__builtins__": builtins}
        try:
            exec(code_obj, g)
            # A common pattern is ``threading.Thread(target=main).start()``: wait for non-daemon threads like the
            # interpreter would at shutdown, otherwise os._exit() below discards their output.
            for t in threading.enumerate():
                if t is not threading.main_thread() and not t.daemon:
                    t.join()
            rc = 0
        except SystemExit as e:
            rc = 0 if e.code in (None, 0) else 1
        except BaseException as e:
            tb = e.__traceback__.tb_next if e.__traceback__ is not None else None  # drop this sandbox frame
            traceback.print_exception(type(e), e, tb)
            rc = 1
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:
            rc = 1
    finally:
        os._exit(rc)


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_one(code_obj, inp: str, timeout: float, out_limit: int, mem_bytes: int) -> tuple[str, str, float, float]:
    """Run compiled ``code_obj`` on one input in a forked sandboxed child; returns (status, out, wall_s, cpu_s)."""
    work_dir = tempfile.mkdtemp(prefix="selectrl_exec_")
    files = [tempfile.TemporaryFile(dir=work_dir) for _ in range(3)]
    inf, outf, errf = files
    try:
        inf.write(inp.encode("utf-8", "replace"))
        inf.flush()
        inf.seek(0)
        cpu_soft = max(1, math.ceil(timeout + 0.5))
        hang_s = 3.0 * timeout + 2.0
        t0 = time.monotonic()
        pid = os.fork()
        if pid == 0:
            _child(code_obj, (inf.fileno(), outf.fileno(), errf.fileno()), work_dir, cpu_soft, mem_bytes)
        delay, hung = 0.0005, False
        while True:
            r, st, ru = os.wait4(pid, os.WNOHANG)
            if r:
                break
            if time.monotonic() - t0 > hang_s:
                _kill_group(pid)
                _, st, ru = os.wait4(pid, 0)
                hung = True
                break
            time.sleep(delay)
            delay = min(delay * 2, 0.01)
        wall = time.monotonic() - t0
        _kill_group(pid)
        cpu = ru.ru_utime + ru.ru_stime
        if hung or cpu > timeout:
            return "timeout", "", wall, cpu
        if os.WIFSIGNALED(st):
            sig = os.WTERMSIG(st)
            if sig == signal.SIGXCPU or (sig == signal.SIGKILL and cpu >= cpu_soft):
                return "timeout", "", wall, cpu
            errf.seek(0)
            err = errf.read()[-500:].decode("utf-8", "replace")
            what = "output file size limit exceeded" if sig == signal.SIGXFSZ else f"killed by signal {sig}"
            return "error", f"{what}\n{err}", wall, cpu
        if os.WEXITSTATUS(st) != 0:
            errf.seek(0)
            return "error", errf.read()[-600:].decode("utf-8", "replace"), wall, cpu
        outf.seek(0)
        data = outf.read(out_limit + 1)
        if len(data) > out_limit:
            return "error", f"output limit of {out_limit} bytes exceeded", wall, cpu
        return "ok", data.decode("utf-8", "replace"), wall, cpu
    finally:
        for f in files:
            f.close()
        shutil.rmtree(work_dir, ignore_errors=True)


def handle(job: dict, mem_bytes: int) -> list[list]:
    """Execute one job and return its result rows."""
    inputs = job["inputs"]
    timeout = float(job.get("timeout", 6.0))
    expected = job.get("expected")
    early_stop = bool(job.get("early_stop", False))
    keep = bool(job.get("keep_out", False))
    key_only = bool(job.get("key_only", False))
    out_limit = min(int(job.get("out_limit", 1 << 20)), FSIZE_LIMIT)
    try:
        code_obj = compile(job["code"], "prog.py", "exec")
    except BaseException:
        msg = traceback.format_exc(limit=0)[-600:]
        return [["compile_error", msg, 0.0, 0.0] for _ in inputs]
    res = []
    for k, inp in enumerate(inputs):
        st, out, wall, cpu = run_one(code_obj, inp, timeout, out_limit, mem_bytes)
        if st == "ok" and expected is not None and k < len(expected) and expected[k] is not None:
            st = "pass" if same_output(out, expected[k]) else "wrong"
        if st in ("ok", "pass", "wrong") and key_only:
            out = out_key(out)
        elif st in ("ok", "pass") and not keep:
            out = ""
        elif st == "wrong" and not keep:
            out = out[:WRONG_HEAD]
        res.append([st, out, round(wall, 4), round(cpu, 4)])
        if early_stop and st not in ("ok", "pass"):
            break
    return res


def main() -> None:
    """Serve jobs from stdin until EOF."""
    for var in THREAD_ENV_VARS:
        os.environ[var] = "1"
    for name in PRELOAD_MODULES:
        try:
            importlib.import_module(name)
        except Exception:  # NumPy is optional
            pass
    mem_bytes = int(os.environ.get("SELECTRL_EXEC_MEM_MB", DEFAULT_MEM_MB)) << 20
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    for line in stdin:
        if not line.strip():
            continue
        job: dict = {}
        try:
            job = json.loads(line)
            res = handle(job, mem_bytes)
        except BaseException as e:
            res = [["error", "worker failure " + repr(e)[:200], 0.0, 0.0]] * max(1, len(job.get("inputs", ())))
        stdout.write(json.dumps({"res": res}).encode("utf-8") + b"\n")
        stdout.flush()


if __name__ == "__main__":
    main()

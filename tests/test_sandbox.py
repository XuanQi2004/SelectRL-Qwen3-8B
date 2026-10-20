"""Sandbox: status codes, CPU-time limits, hang guard, resource limits, NumPy in the sandbox, output comparison."""
from __future__ import annotations

import importlib.util
import sys
import time

import pytest

from selectrl.sandbox import ExecPool, default_workers, out_key, pmap, run_prog, same_output
from selectrl.sandbox.pool import worker_command

ADD = "a, b = map(int, input().split())\nprint(a + b)\n"


def test_status_codes():
    (st, out, wall, cpu), = run_prog(ADD, ["1 2\n"], 2.0)
    assert (st, out) == ("ok", "") and 0.0 <= cpu <= wall < 1.0
    res = run_prog(ADD, ["1 2\n", "3 4\n"], 2.0, expected=["3\n", "8"])
    assert [r[0] for r in res] == ["pass", "wrong"]
    assert res[1][1] == "7\n"  # head of the wrong output
    assert run_prog(ADD, ["1 2\n"], 2.0, keep_out=True)[0][1] == "3\n"
    err = run_prog("raise ValueError('boom')", ["", ""], 2.0)
    assert [r[0] for r in err] == ["error", "error"] and "boom" in err[0][1]
    assert "selectrl" not in err[0][1]  # the sandbox frame is not part of the traceback
    assert [r[0] for r in run_prog("def f(:\n", ["", ""], 2.0)] == ["compile_error"] * 2
    assert [r[0] for r in run_prog(None, ["", ""], 2.0)] == ["nocode"] * 2
    assert run_prog("   \n", ["x"], 2.0)[0][0] == "nocode"
    assert run_prog(ADD, [], 2.0) == []
    assert run_prog("import sys\nsys.exit(0)", [""], 2.0)[0][0] == "ok"
    assert run_prog("import sys\nsys.exit(3)", [""], 2.0)[0][0] == "error"


def test_early_stop_and_key_only():
    res = run_prog(ADD, ["1 2\n", "1 1\n", "5 5\n"], 2.0, expected=["3", "3", "10"], early_stop=True)
    assert [r[0] for r in res] == ["pass", "wrong"]
    res = run_prog("print(' 1.0000001 ', 7)", ["", ""], 2.0, expected=["1 7", None], key_only=True)
    assert [r[0] for r in res] == ["pass", "ok"]
    assert res[0][1] == res[1][1] == out_key("1 7")


def test_cpu_time_verdicts():
    burn = "import time\nt = time.process_time()\nwhile time.process_time() - t < 0.6:\n    pass\nprint(1)\n"
    st, _, wall, cpu = run_prog(burn, [""], 0.25)[0]
    assert st == "timeout" and cpu >= 0.25  # finished, but used more CPU than allowed
    assert run_prog(burn, [""], 2.0)[0][0] == "ok"
    t0 = time.monotonic()
    st, _, _, cpu = run_prog("while True:\n    pass\n", [""], 0.3)[0]
    assert st == "timeout" and cpu >= 0.3 and time.monotonic() - t0 < 5  # killed by RLIMIT_CPU after ~1 s


def test_wall_hang_guard():
    t0 = time.monotonic()
    st, _, wall, cpu = run_prog("import time\ntime.sleep(60)\n", [""], 0.1)[0]
    assert st == "timeout" and cpu < 0.1 and 2.0 <= wall < 5.0  # 3 x limit + 2 s, not 60 s
    assert time.monotonic() - t0 < 6


def test_limits_and_isolation():
    st, detail, _, _ = run_prog("print('x' * 5000)", [""], 2.0, out_limit=1000)[0]
    assert st == "error" and "output limit" in detail
    st, detail, _, _ = run_prog("import socket\nsocket.socket()\n", [""], 2.0)[0]
    assert st == "error" and "network access is disabled" in detail
    write = "open('f.txt', 'w').write('x')\nimport os, tempfile\nprint(os.listdir('.'), os.path.realpath(tempfile.gettempdir()) == os.getcwd())"
    first = run_prog(write, ["", ""], 2.0, keep_out=True)
    assert all(r[0] == "ok" and r[1].count("f.txt") == 1 and "True" in r[1] for r in first)  # fresh dir per run
    st, detail, _, _ = run_prog("open('big', 'wb').write(b'x' * (80 << 20))", [""], 5.0)[0]
    assert st == "error" and "File too large" in detail  # RLIMIT_FSIZE 64 MiB (Python ignores SIGXFSZ: EFBIG)
    threaded = "import threading\ndef main():\n    print(input())\nthreading.Thread(target=main).start()\n"
    assert run_prog(threaded, ["hello\n"], 2.0, keep_out=True)[0][:2] == ("ok", "hello\n")
    env = "import os\nprint(os.environ['OMP_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS'], os.environ['MKL_NUM_THREADS'])"
    assert run_prog(env, [""], 2.0, keep_out=True)[0][1].split() == ["1", "1", "1"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is not enforceable on macOS")
def test_memory_limit():
    st, _, _, _ = run_prog("x = bytearray(3 << 30)\nprint(len(x))\n", [""], 5.0)[0]
    assert st == "error"


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="numpy not installed")
def test_numpy_in_sandbox_is_fast():
    code = "import numpy as np\nprint(int(np.arange(int(input())).sum()))\n"
    run_prog(code, ["1\n"], 5.0)  # make sure a worker is up
    res = run_prog(code, ["10\n", "1000\n"], 5.0, keep_out=True)
    assert [r[:2] for r in res] == [("ok", "45\n"), ("ok", "499500\n")]
    assert all(r[2] < 1.0 for r in res)  # preloaded in the worker: no per-run import cost


def test_worker_crash_is_contained():
    pool = ExecPool(n_workers=1)
    try:
        kill_parent = "import os, signal\nos.kill(os.getppid(), signal.SIGKILL)\n"
        job = {"code": kill_parent, "inputs": [""], "timeout": 2.0}
        assert pool.run(job)[0][0] == "error"
        assert pool.run({"code": ADD, "inputs": ["2 2\n"], "timeout": 2.0, "keep_out": True})[0][:2] == ["ok", "4\n"]
    finally:
        pool.close()


def test_pool_configuration(monkeypatch):
    monkeypatch.setenv("SELECTRL_EXEC_WORKERS", "7")
    assert default_workers() == 7
    monkeypatch.delenv("SELECTRL_EXEC_WORKERS")
    assert 4 <= default_workers() <= 256
    monkeypatch.setenv("SELECTRL_UNSHARE_NET", "0")
    assert worker_command()[0] == sys.executable
    assert pmap(lambda x: x * x, range(20), workers=4) == [x * x for x in range(20)]


def test_same_output():
    assert same_output("1 2\n3\n", "1 2\n3")
    assert same_output("  1 2   \n3  \n\n", "1 2\n3\n")
    assert same_output("1 2 3", "1\n2\n3")  # token fallback ignores line structure
    assert same_output("0.3333333", "0.333333")
    assert same_output("1000000.5", "1000000.0")  # relative tolerance 1e-6
    assert not same_output("0.334", "0.333")
    assert not same_output("1 2", "1 2 3")
    assert not same_output("YES", "NO")
    assert not same_output("inf", "1e400")


def test_out_key():
    assert out_key("  1.234561   \n\t 42 ") == out_key("1.234562 42")
    assert out_key("1.23456 42") != out_key("42 1.23456")
    assert out_key("-0.0") == out_key("0.000")
    assert out_key("1e3") == out_key("1000.0")
    assert out_key("2.0") == out_key("2") != out_key("02")  # integer tokens are kept verbatim
    assert out_key("YES") != out_key("yes")
    assert out_key(None) == out_key("") == out_key(" \n ")
    assert len(out_key("x")) == 16

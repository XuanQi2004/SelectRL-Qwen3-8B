"""Launch and supervise one vLLM OpenAI-compatible server per GPU.

Usage (normally called by scripts/*.sh, which keep this process in the background):

    python -m selectrl.rollout.servers launch --gpus 0,1,2,3 --mem-util 0.40 --state runs/x/servers.json \
        --log-dir runs/x/logs --model models/Qwen3-8B

The launcher starts the servers, waits until every server answers ``/v1/models``, writes the state file
(``{"ready": true, "servers": [{"gpu": 0, "url": "http://127.0.0.1:18000", ...}]}``) atomically and then supervises
them: if any server dies it rewrites the state file with ``"ready": false`` and the reason, stops the others and
exits with code 3. SIGTERM / SIGINT stop all servers (only the process groups this launcher created).

Each server runs with a clean environment: ``CUDA_VISIBLE_DEVICES`` is set to its GPU and every torchrun /
torch.distributed variable is removed, so a server can never join the trainer's process group.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from selectrl.utils import atomic_write_json, log

_DIST_ENV = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "OMP_NUM_THREADS",
)


def _free_port(start: int, avoid: set) -> int:
    port = start
    while True:
        if port not in avoid:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    pass
        port += 1


def server_command(args: argparse.Namespace, port: int) -> List[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--served-model-name", "base",
        "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", f"{args.mem_util:.3f}",
        "--max-num-seqs", str(args.max_num_seqs),
        "--enable-prefix-caching",
        "--generation-config", "vllm",
        "--seed", str(args.seed),
        "--uvicorn-log-level", "warning",
    ]
    if args.language_model_only:
        cmd.append("--language-model-only")
    if args.lora_rank > 0:
        cmd += ["--enable-lora", "--max-lora-rank", str(args.lora_rank), "--max-loras", "2"]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.extra:
        cmd += args.extra.split()
    return cmd


def _server_env(gpu: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _DIST_ENV}
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    # many-core hosts: cap per-server CPU thread pools (8 servers x 384 threads would oversubscribe the host)
    env["OMP_NUM_THREADS"] = os.environ.get("SELECTRL_SERVER_THREADS", "8")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", os.environ.get("SELECTRL_SERVER_THREADS", "8"))
    for k in ("no_proxy", "NO_PROXY"):
        env[k] = "127.0.0.1,localhost" + ("," + env[k] if env.get(k) else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy


def _ready(url: str) -> bool:
    try:
        with _NO_PROXY_OPENER.open(url + "/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    keep = [l for l in lines if "Capturing CUDA graphs" not in l]
    return "\n".join(keep[-n:])


def launch(args: argparse.Namespace) -> int:
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)
    procs: List[subprocess.Popen] = []
    servers: List[dict] = []
    used: set = set()

    def stop_all(*_a) -> None:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.time() + 30
        for p in procs:
            try:
                p.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def on_signal(signum, _frame) -> None:
        log(f"[servers] signal {signum}: stopping {len(procs)} server(s)")
        stop_all()
        atomic_write_json(state_path, {"ready": False, "servers": servers, "reason": f"stopped by signal {signum}"})
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    atomic_write_json(state_path, {"ready": False, "servers": [], "reason": "starting"})
    for gpu in gpus:
        port = _free_port(args.port_base + int(gpu) * 7 if gpu.isdigit() else args.port_base, used)
        used.add(port)
        lf = log_dir / f"vllm_gpu{gpu}.log"
        fh = open(lf, "ab")
        p = subprocess.Popen(server_command(args, port), env=_server_env(gpu), stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        procs.append(p)
        servers.append({"gpu": gpu, "port": port, "url": f"http://127.0.0.1:{port}", "pid": p.pid,
                        "log": str(lf)})
        log(f"[servers] gpu {gpu}: pid {p.pid} port {port} log {lf}")
        if args.stagger > 0:
            time.sleep(args.stagger)

    t0 = time.time()
    pending = set(range(len(servers)))
    while pending:
        for i in list(pending):
            if procs[i].poll() is not None:
                msg = f"vLLM server on GPU {servers[i]['gpu']} exited with code {procs[i].returncode} during startup"
                log(f"[servers] ERROR {msg}\n--- last log lines ({servers[i]['log']}):\n{_tail(Path(servers[i]['log']))}")
                stop_all()
                atomic_write_json(state_path, {"ready": False, "servers": servers, "reason": msg})
                return 3
            if _ready(servers[i]["url"]):
                pending.discard(i)
                log(f"[servers] gpu {servers[i]['gpu']} ready after {time.time() - t0:.0f}s")
        if time.time() - t0 > args.startup_timeout:
            log(f"[servers] ERROR startup timeout ({args.startup_timeout}s); pending GPUs "
                f"{[servers[i]['gpu'] for i in pending]}")
            stop_all()
            atomic_write_json(state_path, {"ready": False, "servers": servers, "reason": "startup timeout"})
            return 3
        time.sleep(3)

    atomic_write_json(state_path, {"ready": True, "servers": servers, "started": time.time(),
                                   "model": args.model, "max_model_len": args.max_model_len,
                                   "mem_util": args.mem_util})
    log(f"[servers] all {len(servers)} server(s) ready in {time.time() - t0:.0f}s -> {state_path}")
    while True:
        for i, p in enumerate(procs):
            if p.poll() is not None:
                msg = f"vLLM server on GPU {servers[i]['gpu']} died with code {p.returncode}"
                log(f"[servers] ERROR {msg}\n{_tail(Path(servers[i]['log']))}")
                stop_all()
                atomic_write_json(state_path, {"ready": False, "servers": servers, "reason": msg})
                return 3
        time.sleep(5)


def wait_ready(state_path: str | Path, timeout: float = 3600.0, launcher_pid: Optional[int] = None) -> dict:
    """Block until the launcher reports ready (returns the state) or failed (raises RuntimeError)."""
    t0 = time.time()
    p = Path(state_path)
    while True:
        if p.exists():
            try:
                st = json.loads(p.read_text())
            except json.JSONDecodeError:
                st = {}
            if st.get("ready"):
                return st
            if st.get("reason") not in (None, "starting"):
                raise RuntimeError(f"vLLM servers failed: {st.get('reason')}")
        if launcher_pid is not None:
            try:
                os.kill(launcher_pid, 0)
            except ProcessLookupError:
                raise RuntimeError("vLLM server launcher exited before the servers became ready")
        if time.time() - t0 > timeout:
            raise RuntimeError(f"timed out after {timeout:.0f}s waiting for {p}")
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch", help="start one vLLM server per GPU and supervise them")
    la.add_argument("--gpus", required=True, help="comma-separated physical GPU indices, e.g. 0,1,2,3")
    la.add_argument("--model", required=True)
    la.add_argument("--state", required=True, help="state JSON written when ready")
    la.add_argument("--log-dir", required=True)
    la.add_argument("--mem-util", type=float, default=0.40)
    la.add_argument("--max-model-len", type=int, default=40960)
    la.add_argument("--max-num-seqs", type=int, default=256)
    la.add_argument("--lora-rank", type=int, default=64, help="0 disables LoRA")
    la.add_argument("--port-base", type=int, default=int(os.environ.get("SELECTRL_PORT_BASE", "18000")))
    la.add_argument("--seed", type=int, default=0)
    la.add_argument("--language-model-only", type=int, default=1)
    la.add_argument("--enforce-eager", action="store_true")
    la.add_argument("--stagger", type=float, default=2.0, help="seconds between server starts")
    la.add_argument("--startup-timeout", type=float, default=2400.0)
    la.add_argument("--extra", default=os.environ.get("SELECTRL_VLLM_EXTRA", ""), help="extra vLLM CLI flags")
    wa = sub.add_parser("wait", help="wait until a launcher state file says ready")
    wa.add_argument("--state", required=True)
    wa.add_argument("--timeout", type=float, default=3600.0)
    wa.add_argument("--pid", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "launch":
        sys.exit(launch(args))
    st = wait_ready(args.state, args.timeout, args.pid)
    print(json.dumps(st))


if __name__ == "__main__":
    main()

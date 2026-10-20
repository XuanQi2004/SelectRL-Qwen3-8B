"""Generate the helper programs of the execution checker with the frozen base model (thinking off).

Ported from SSPO v1 (ec_gen.py); the offline ``vllm.LLM`` engine is replaced by the same per-GPU vLLM servers that
serve rollouts, reached through an async ``sampler`` (``selectrl.rollout.client.ServerPool`` or ``MockSampler``):

    async sample(prompt_ids, n, max_tokens, temperature, top_p, top_k, seed, model=None) -> [{"text", "token_ids",
    "finish_reason"}, ...]

and a ``render_chat(messages) -> list[int]`` callable that applies the model's chat template (thinking off).
Per problem, only the statement is shown: 4 input generators (``gen``), 4 brute-force solutions (``bf``) and 2 input
validators (``val``). Results are written per problem to ``<cache>/checkers/artifacts/<pid>.json.gz`` with atomic
writes; existing files are skipped, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from selectrl.utils import atomic_write_json_gz, log, sha1_16

from .chk import last_code_block

KINDS: dict[str, int] = {"gen": 4, "bf": 4, "val": 2}
SAMPLING: dict[str, Any] = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 262144}  # clamped to the context window by the client

HEAD = (
    "Below is a competitive programming problem (standard input / standard output).\n\n"
    "[PROBLEM]\n{q}\n[END OF PROBLEM]\n\n"
)
PROMPTS: dict[str, str] = {
    "gen": HEAD + (
        "Write a Python 3 program that generates ONE valid test input for this problem.\n"
        "The program reads two values from standard input: MODE and SEED (for example `max 3`). Call random.seed(SEED).\n"
        "MODE is one of:\n"
        "- max: an input as large as the constraints allow (maximum sizes and value ranges), designed to be a worst case for "
        "running time, e.g. structures that make naive or quadratic algorithms slow;\n"
        "- small: a small random valid input (tiny sizes, e.g. at most 6 elements, and small values) that a brute-force "
        "solution can handle;\n"
        "- edge: a small input that exercises boundary cases (minimum sizes, all values equal, extreme values, and similar).\n"
        "The generated input must strictly satisfy every constraint and the exact input format of the statement, including "
        "any guarantees the statement makes about the input. Print only the input, nothing else.\n"
        "Give the complete program in a single ```python code block."
    ),
    "bf": HEAD + (
        "Write a brute-force Python 3 solution that is obviously correct, intended only for very small inputs. It may be "
        "exponentially slow. Prefer directly enumerating all possibilities exactly as the statement defines them over any "
        "clever algorithm, so that its correctness is easy to trust. Read from standard input and print the answer in the "
        "required output format.\nGive the complete program in a single ```python code block."
    ),
    "val": HEAD + (
        "Write a Python 3 input validator. It reads one test input from standard input and prints exactly VALID if the "
        "input follows the input format and satisfies every constraint and guarantee in the statement, otherwise exactly "
        "INVALID. Be strict about the number of tokens and lines and about all value ranges.\n"
        "Give the complete program in a single ```python code block."
    ),
}


class Sampler(Protocol):
    """Async completion sampler over token ids."""

    async def sample(self, prompt_ids: list[int], n: int, max_tokens: int, temperature: float, top_p: float,
                     top_k: int, seed: int, model: str | None = None) -> list[dict]: ...


def helper_messages(kind: str, question: str) -> list[dict[str, str]]:
    """Chat messages of one helper request."""
    return [{"role": "user", "content": PROMPTS[kind].format(q=question)}]


def helper_seed(pid: str, kind: str) -> int:
    """Sampling seed of a (problem, kind) request: ``(int(sha1_16(pid), 16) * 10 + kind index) mod 2^31``."""
    return (int(sha1_16(pid), 16) * 10 + list(KINDS).index(kind)) % (2**31)


def artifact_path(artifacts_dir: str | Path, pid: str) -> Path:
    """Path of a problem's artifacts file."""
    return Path(artifacts_dir) / f"{pid}.json.gz"


async def generate_problem_artifacts(problem: dict[str, Any], sampler: Sampler, render_chat: Callable[[list], list[int]],
                                     model: str | None = None) -> dict[str, Any]:
    """Sample the helper programs of one problem (three concurrent requests)."""
    kinds = list(KINDS)
    outs = await asyncio.gather(*(
        sampler.sample(render_chat(helper_messages(kind, problem["question"])), n=KINDS[kind],
                       seed=helper_seed(problem["pid"], kind), model=model, **SAMPLING)
        for kind in kinds
    ))
    art: dict[str, Any] = {"pid": problem["pid"]}
    for kind, choices in zip(kinds, outs):
        art[kind] = [last_code_block(c["text"]) for c in choices]
        art[kind + "_meta"] = [{"finish": c.get("finish_reason"), "tokens": len(c.get("token_ids") or ())}
                               for c in choices]
    return art


async def generate_artifacts(problems: Iterable[dict[str, Any]], artifacts_dir: str | Path, sampler: Sampler,
                             render_chat: Callable[[list], list[int]], model: str | None = None,
                             concurrency: int = 64) -> dict[str, Any]:
    """Generate and persist the artifacts of every problem that has none yet, ``concurrency`` problems at a time.

    A failed problem is logged and left without a file, so the next run retries it.
    """
    probs = list(problems)
    todo = [p for p in probs if not artifact_path(artifacts_dir, p["pid"]).exists()]
    log(f"[gen] problems {len(probs)}, to generate {len(todo)}")
    sem = asyncio.Semaphore(max(1, concurrency))
    failed: dict[str, str] = {}
    n_code = dict.fromkeys(KINDS, 0)

    async def one(problem: dict[str, Any]) -> None:
        async with sem:
            try:
                art = await generate_problem_artifacts(problem, sampler, render_chat, model)
            except Exception as e:
                failed[problem["pid"]] = repr(e)[:300]
                log(f"[gen] {problem['pid']} FAILED {failed[problem['pid']]}")
                return
        await asyncio.to_thread(atomic_write_json_gz, artifact_path(artifacts_dir, problem["pid"]), art)
        for kind in KINDS:
            n_code[kind] += sum(c is not None for c in art[kind])

    await asyncio.gather(*(one(p) for p in todo))
    summary = {"n_problems": len(probs), "generated": len(todo) - len(failed), "reused": len(probs) - len(todo),
               "failed": failed, "code_blocks": n_code}
    log(f"[gen] {summary}")
    return summary


# ---------------------------------------------------------------- deterministic mock for tests


def mock_render_chat(messages: Sequence[Mapping[str, str]]) -> list[int]:
    """Byte-level stand-in for a chat template: UTF-8 bytes of ``<role>\\ncontent\\n`` blocks."""
    return list("".join(f"<{m['role']}>\n{m['content']}\n" for m in messages).encode("utf-8"))


def prompt_kind(prompt: str) -> str | None:
    """Which helper kind a rendered prompt asks for."""
    for kind, text in PROMPTS.items():
        if text.split("[END OF PROBLEM]\n\n", 1)[1] in prompt:
            return kind
    return None


class MockSampler:
    """Deterministic sampler answering helper prompts from a table ``{question: {kind: [program, ...]}}``.

    Choice ``i`` of a request is the ``i mod len``-th listed program in a fenced block; unknown questions or kinds
    get an answer without code. Expects prompts rendered by ``mock_render_chat``.
    """

    def __init__(self, programs: Mapping[str, Mapping[str, Sequence[str]]]) -> None:
        self.programs = programs
        self.calls: list[dict[str, Any]] = []

    async def sample(self, prompt_ids: list[int], n: int, max_tokens: int, temperature: float, top_p: float,
                     top_k: int, seed: int, model: str | None = None) -> list[dict]:
        prompt = bytes(prompt_ids).decode("utf-8")
        question = prompt.split("[PROBLEM]\n", 1)[-1].split("\n[END OF PROBLEM]", 1)[0]
        kind = prompt_kind(prompt)
        self.calls.append({"kind": kind, "question": question, "n": n, "seed": seed, "max_tokens": max_tokens})
        codes = list(self.programs.get(question, {}).get(kind or "", ()))
        texts = [f"Here is the program.\n```python\n{codes[i % len(codes)].rstrip()}\n```\n" if codes else "No program."
                 for i in range(n)]
        return [{"text": t, "token_ids": list(t.encode("utf-8")), "finish_reason": "stop"} for t in texts]


# ---------------------------------------------------------------- command line (uses the running vLLM servers)


def main() -> None:
    """python -m selectrl.checker.gen --servers runs/_chk/servers.json --problems data/problems/val.jsonl.gz"""
    import argparse

    from selectrl.config import load_protocol, model_path, resolve
    from selectrl.rollout.client import ServerPool
    from selectrl.rollout.prompts import ChatRenderer
    from selectrl.utils import iter_jsonl

    ap = argparse.ArgumentParser(description="Generate the checker helper programs with the base model (thinking off).")
    ap.add_argument("--servers", required=True, help="state file of selectrl.rollout.servers launch")
    ap.add_argument("--problems", required=True, nargs="+", help="problems .jsonl.gz files")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--protocol", default="configs/protocol/qwen3_8b_40960.yaml")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    cfg = load_protocol(a.protocol)
    render = ChatRenderer(model_path(cfg), enable_thinking=False)
    pool = ServerPool.from_state(resolve(a.servers))
    probs = sorted((p for f in a.problems for p in iter_jsonl(resolve(f))), key=lambda p: p["pid"])
    probs = probs[a.shard::a.nshards]
    out = resolve(a.cache) / "checkers" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(generate_artifacts(probs, out, pool, render, concurrency=a.concurrency))


if __name__ == "__main__":
    main()

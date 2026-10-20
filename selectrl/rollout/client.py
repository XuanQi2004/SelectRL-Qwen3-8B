"""Asynchronous sampling client for the per-GPU vLLM servers.

All prompts are sent as token ids to ``/v1/completions`` so that the trainer sees exactly the sampled tokens
(``return_token_ids``) and their sampling-time log-probabilities (``logprobs: 0``). Requests are spread over the
servers by least outstanding work; each server receives at most ``max_inflight`` concurrent requests.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from selectrl.utils import log


@dataclass
class SampleRequest:
    prompt_ids: List[int]
    n: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    model: Optional[str] = None  # adapter name; None -> base model
    tag: Any = None  # caller bookkeeping (returned unchanged)


def parse_choice(choice: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one completion choice to {text, token_ids, logprobs, finish_reason}."""
    token_ids = choice.get("token_ids")
    lp = choice.get("logprobs") or {}
    token_logprobs = lp.get("token_logprobs")
    if token_ids is None and lp.get("tokens") and str(lp["tokens"][0]).startswith("token_id:"):
        token_ids = [int(str(t).split(":", 1)[1]) for t in lp["tokens"]]
    if token_ids is None:
        raise RuntimeError("vLLM response has no token ids; the server must support return_token_ids")
    if token_logprobs is None:
        token_logprobs = [0.0] * len(token_ids)
    token_logprobs = [float(x) if x is not None else 0.0 for x in token_logprobs]
    if len(token_logprobs) != len(token_ids):
        n = min(len(token_logprobs), len(token_ids))
        token_ids, token_logprobs = token_ids[:n], token_logprobs[:n]
    return {"text": choice.get("text", ""), "token_ids": list(map(int, token_ids)), "logprobs": token_logprobs,
            "finish_reason": choice.get("finish_reason")}


class ServerPool:
    """Client-side view of the running servers (read from the launcher's state file)."""

    def __init__(self, urls: Sequence[str], max_inflight: int = 64, request_timeout: float = 7200.0,
                 retries: int = 4, max_model_len: int = 40960):
        if not urls:
            raise ValueError("no vLLM servers")
        self.urls = list(urls)
        self.max_inflight = max_inflight
        self.request_timeout = request_timeout
        self.retries = retries
        self._load = [0] * len(self.urls)
        self.max_model_len = int(max_model_len)
        self.adapters: Dict[str, str] = {}

    @classmethod
    def from_state(cls, state_path: str | Path, **kw) -> "ServerPool":
        st = json.loads(Path(state_path).read_text())
        if not st.get("ready"):
            raise RuntimeError(f"servers not ready: {st.get('reason')}")
        kw.setdefault("max_model_len", int(st.get("max_model_len", 40960)))
        return cls([s["url"] for s in st["servers"]], **kw)

    # ------------------------------------------------------------------ sampling
    async def _post(self, session, url: str, path: str, payload: dict) -> dict:
        import aiohttp

        last = None
        for attempt in range(self.retries):
            try:
                async with session.post(url + path, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as r:
                    body = await r.text()
                    if r.status != 200:
                        raise RuntimeError(f"HTTP {r.status} from {url}{path}: {body[:500]}")
                    return json.loads(body)
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:  # noqa: PERF203
                last = e
                if isinstance(e, RuntimeError) and "HTTP 4" in str(e):
                    raise
                await asyncio.sleep(2.0 * (attempt + 1) + random.random())
        raise RuntimeError(f"request to {url}{path} failed after {self.retries} attempts: {last}")

    async def _sample_one(self, session, sems, req: SampleRequest) -> List[Dict[str, Any]]:
        i = min(range(len(self.urls)), key=lambda j: self._load[j])
        work = req.n * (len(req.prompt_ids) // 8 + req.max_tokens // 64 + 1)
        self._load[i] += work
        try:
            async with sems[i]:
                payload = {
                    "model": req.model or "base",
                    "prompt": req.prompt_ids,
                    "n": req.n,
                    # never cap below the context window: prompt + response may use all of max_model_len
                    "max_tokens": max(1, min(req.max_tokens, self.max_model_len - len(req.prompt_ids))),
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "top_k": req.top_k,
                    "seed": req.seed,
                    "logprobs": 0,
                    "return_token_ids": True,
                    "skip_special_tokens": True,
                }
                out = await self._post(session, self.urls[i], "/v1/completions", payload)
        finally:
            self._load[i] -= work
        choices = sorted(out["choices"], key=lambda c: c.get("index", 0))
        return [parse_choice(c) for c in choices]

    async def _sample_all(self, reqs: Sequence[SampleRequest], progress_every: float) -> List[List[Dict[str, Any]]]:
        import aiohttp

        sems = [asyncio.Semaphore(self.max_inflight) for _ in self.urls]
        conn = aiohttp.TCPConnector(limit=0, force_close=True)
        done = 0
        t0 = time.time()
        last = t0
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = [asyncio.create_task(self._sample_one(session, sems, r)) for r in reqs]
            for fut in asyncio.as_completed(tasks):
                await fut
                done += 1
                if progress_every and time.time() - last > progress_every:
                    last = time.time()
                    log(f"[sample] {done}/{len(reqs)} requests done ({time.time() - t0:.0f}s)")
            return [t.result() for t in tasks]

    def sample_many(self, reqs: Sequence[SampleRequest], progress_every: float = 60.0) -> List[List[Dict[str, Any]]]:
        """Run all requests concurrently; returns, per request, its ``n`` parsed choices in index order."""
        if not reqs:
            return []
        return asyncio.run(self._sample_all(reqs, progress_every))

    async def sample(self, prompt_ids: List[int], n: int, max_tokens: int, temperature: float, top_p: float,
                     top_k: int, seed: int, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Single request (used by the checker helper generator, which batches its own calls)."""
        import aiohttp

        sems = [asyncio.Semaphore(self.max_inflight) for _ in self.urls]
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True)) as s:
            return await self._sample_one(s, sems, SampleRequest(prompt_ids, n, max_tokens, temperature, top_p,
                                                                 top_k, seed, model))

    # ------------------------------------------------------------------ adapters
    async def _all(self, path: str, payload: dict, ok_substrings: Sequence[str] = ()) -> None:
        import aiohttp

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True)) as session:
            async def one(url: str) -> None:
                async with session.post(url + path, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as r:
                    body = await r.text()
                    if r.status != 200 and not any(s in body for s in ok_substrings):
                        raise RuntimeError(f"{path} on {url} failed: HTTP {r.status} {body[:400]}")
            await asyncio.gather(*(one(u) for u in self.urls))

    def load_adapter(self, name: str, path: str | Path) -> None:
        asyncio.run(self._all("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(Path(path).resolve())},
                              ok_substrings=("already been loaded",)))
        self.adapters[name] = str(path)

    def unload_adapter(self, name: str) -> None:
        asyncio.run(self._all("/v1/unload_lora_adapter", {"lora_name": name}, ok_substrings=("not found", "cannot be found")))
        self.adapters.pop(name, None)

    def swap_adapter(self, new_name: str, new_path: str | Path, old_name: Optional[str]) -> None:
        """Load the new adapter first, then unload the old one (``--max-loras 2`` keeps both resident briefly)."""
        self.load_adapter(new_name, new_path)
        if old_name and old_name != new_name:
            try:
                self.unload_adapter(old_name)
            except RuntimeError as e:  # unloading is best effort
                log(f"[servers] warning: could not unload adapter {old_name}: {e}")

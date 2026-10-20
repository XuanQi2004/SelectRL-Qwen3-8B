"""Execution checker ``Chk``, execution signature ``sig`` and hidden-test label ``g`` of one program.

Ported from SSPO v1 (checker.py evaluate_chk_and_sig, ec_run.py run_candidate_on_cases, es_label.py
label_program_on_hidden, ec_common.py norm_code / code_hash / last_code_block).

Definitions (cases from ``selectrl.checker.build``; all runs use the problem's ``time_limit_s`` of CPU time):
  Chk(y) = 1 iff (1) y passes every public example, (2) y finishes with status ok on every valid or unverified
           stress (``max``) input, and (3) y finishes ok with the reference output key on every confirmed
           reference input (``ref_agree >= 2``).
  sig(y) = SHA-1[:16] of the space-joined output keys (``out_key``) of y on all usable check inputs in case order
           (``ERR:<status>`` / ``ST:<status>`` for failed runs); the public output keys if there is no check input.
  g(y)   = 1 iff y passes every hidden test (execution stops at the first failure); g_frac = passed / run.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from selectrl.sandbox import out_key, run_prog
from selectrl.utils import sha1_16

from .build import CHECKER_VERSION

DEFAULT_TIME_LIMIT_S = 6.0
OUT_LIMIT = 16 << 20  # hidden outputs may reach 4 MiB; leave room for formatting differences
NOCODE_KEY = "nocode"
NOCODE_SIG = sha1_16("NOCODE")

_FENCE_RE = re.compile(r"```(?:python3|python|py|Python)?[ \t]*\n(.*?)```", re.S)


# ---------------------------------------------------------------- programs


def strip_think(text: str) -> str:
    """The answer after the last ``</think>``; empty if a ``<think>`` block is never closed (truncated thinking)."""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return "" if "<think>" in text else text


def last_code_block(text: str | None) -> str | None:
    """The last fenced ```python (or untagged) block of a response, after stripping any thinking prefix."""
    if text is None:
        return None
    blocks = _FENCE_RE.findall(strip_think(text))
    if not blocks:
        return None
    code = blocks[-1].strip()
    return code + "\n" if code else None


def norm_code(code: str | None) -> str:
    """Code without blank lines, comment-only lines and trailing spaces (de-duplication key)."""
    if not code:
        return ""
    return "\n".join(line.rstrip() for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


def code_hash(code: str | None) -> str:
    """SHA-1[:16] of ``norm_code(code)``, or ``"nocode"`` when nothing remains."""
    n = norm_code(code)
    return sha1_16(n) if n else NOCODE_KEY


# ---------------------------------------------------------------- execution


def _time_limit(problem: dict[str, Any]) -> float:
    return float(problem.get("time_limit_s", DEFAULT_TIME_LIMIT_S))


def usable_inputs(cases: dict[str, Any], stress: bool) -> list[dict[str, Any]]:
    """Valid or unverified inputs that still carry their content: stress (``max``) or small / edge ones."""
    return [x for x in cases.get("inputs", [])
            if (x.get("mode") == "max") == stress and x.get("valid") is not False and x.get("inp")]


def run_candidate_on_cases(code: str | None, problem: dict[str, Any], cases: dict[str, Any]) -> dict[str, Any]:
    """Run a program on the public examples and the usable check inputs of its problem.

    Returns ``{"public_all", "public_frac", "public_keys", "max": [[id, status, wall_s, cpu_s, key|None]],
    "small": [[id, status, key|None]]}``; stress inputs stop at the first failure.
    """
    if not code or not code.strip():
        return {"public_all": False, "public_frac": 0.0, "public_keys": [], "max": [], "small": []}
    tl = _time_limit(problem)
    ex_in = [t[0] for t in problem.get("public", [])]
    ex_out = [t[1] for t in problem.get("public", [])]
    pub = run_prog(code, ex_in, tl, expected=ex_out, key_only=True, out_limit=OUT_LIMIT)
    n_pass = sum(r[0] == "pass" for r in pub)
    mx, sm = usable_inputs(cases, True), usable_inputs(cases, False)
    rm = run_prog(code, [x["inp"] for x in mx], tl, early_stop=True, key_only=True, out_limit=OUT_LIMIT)
    rs = run_prog(code, [x["inp"] for x in sm], tl, key_only=True, out_limit=OUT_LIMIT)
    return {
        "public_all": n_pass == len(ex_in),
        "public_frac": n_pass / len(ex_in) if ex_in else 1.0,
        "public_keys": [r[1] if r[0] in ("pass", "wrong", "ok") else f"ERR:{r[0]}" for r in pub],
        "max": [[x["id"], r[0], r[2], r[3], r[1] if r[0] == "ok" else None] for x, r in zip(mx, rm)],
        "small": [[x["id"], r[0], r[1] if r[0] == "ok" else None] for x, r in zip(sm, rs)],
    }


def evaluate_chk_and_sig(exec_rec: dict[str, Any] | None, cases: dict[str, Any] | None) -> tuple[int, str]:
    """``(Chk, sig)`` from an execution record of ``run_candidate_on_cases`` and the problem's cases."""
    if not exec_rec or cases is None:
        return 0, NOCODE_SIG
    max_map = {it[0]: (it[1], it[4]) for it in exec_rec.get("max", [])}
    small_map = {it[0]: (it[1], it[2]) for it in exec_rec.get("small", [])}
    confirmed = {x["id"]: x.get("ref_key") or out_key(x["ref"])
                 for x in usable_inputs(cases, False) if x.get("ref") is not None and int(x.get("ref_agree", 0)) >= 2}

    pub_ok = bool(exec_rec.get("public_all", False))
    max_ok = all(max_map.get(x["id"], ("missing",))[0] == "ok" for x in usable_inputs(cases, True))
    ref_ok = all(small_map.get(i, ("missing", None)) == ("ok", k) for i, k in confirmed.items())
    chk = int(pub_ok and max_ok and ref_ok)

    parts: list[str] = []
    for x in cases.get("inputs", []):
        if x.get("valid") is False or not x.get("inp"):
            continue
        if x.get("mode") != "max":
            st, key = small_map.get(x["id"], ("missing", None))
            parts.append(key if st == "ok" and key else f"ERR:{st}")
        else:
            st, key = max_map.get(x["id"], ("missing", None))
            parts.append((key if st == "ok" else f"ERR:{st}") if key is not None else f"ST:{st}")
    if not parts:
        parts = [str(k) for k in exec_rec.get("public_keys", [])] or ["EMPTY"]
    return chk, sha1_16(" ".join(parts))


def label_program_on_hidden(code: str | None, problem: dict[str, Any]) -> dict[str, Any]:
    """Hidden-test label: ``{"all", "frac", "n", "n_run", "fail_type"}`` (stops at the first failing test)."""
    tests = problem.get("hidden") or []
    if not tests:
        return {"all": False, "frac": 0.0, "n": 0, "n_run": 0, "fail_type": "no_tests"}
    if not code or not code.strip():
        return {"all": False, "frac": 0.0, "n": len(tests), "n_run": 0, "fail_type": "nocode"}
    res = run_prog(code, [t[0] for t in tests], _time_limit(problem), expected=[t[1] for t in tests],
                   early_stop=True, key_only=True, out_limit=OUT_LIMIT)
    n_pass = sum(r[0] == "pass" for r in res)
    bad = Counter(r[0] for r in res if r[0] != "pass")
    return {
        "all": n_pass == len(tests),
        "frac": n_pass / max(1, len(res)),
        "n": len(tests),
        "n_run": len(res),
        "fail_type": bad.most_common(1)[0][0] if bad else None,
    }


def nocode_record() -> dict[str, Any]:
    """Execution record of a response without a program."""
    return {"key": NOCODE_KEY, "has_code": 0, "public_pass": 0, "public_frac": 0.0, "chk": 0, "sig": NOCODE_SIG,
            "g": 0, "g_frac": 0.0, "checker_version": CHECKER_VERSION}


def evaluate_program(problem: dict[str, Any], cases: dict[str, Any], code: str | None) -> dict[str, Any]:
    """Execution record of one program: public pass, Chk, sig and hidden label (ARCHITECTURE.md section 3)."""
    key = code_hash(code)
    if key == NOCODE_KEY:
        return nocode_record()
    rec = run_candidate_on_cases(code, problem, cases)
    chk, sig = evaluate_chk_and_sig(rec, cases)
    lab = label_program_on_hidden(code, problem)
    return {
        "key": key,
        "has_code": 1,
        "public_pass": int(rec["public_all"]),
        "public_frac": float(rec["public_frac"]),
        "chk": chk,
        "sig": sig,
        "g": int(lab["all"]),
        "g_frac": float(lab["frac"]),
        "checker_version": CHECKER_VERSION,
    }

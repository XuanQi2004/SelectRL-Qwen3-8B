"""Output comparison: the tolerant judge ``same_output`` and the canonical output key ``out_key``.

Ported from SSPO v1 (exec_worker.py same_output, ec_common.py out_key).

This module is imported both as ``selectrl.sandbox.compare`` and, by the sandbox worker that runs as a plain script,
as ``compare``; it therefore depends on the standard library only.
"""
from __future__ import annotations

import hashlib
import math


def same_output(got: str, exp: str) -> bool:
    """Judge ``got`` against the expected output ``exp``.

    Equal if the line sequences agree after stripping the text and trailing spaces of each line; otherwise the
    whitespace-separated tokens must have the same count and each pair must be identical or two finite floats with
    ``|a - b| <= 1e-6 * max(1, |b|)``.
    """
    g = [line.rstrip() for line in got.strip().splitlines()]
    e = [line.rstrip() for line in exp.strip().splitlines()]
    if g == e:
        return True
    gt, et = got.split(), exp.split()
    if len(gt) != len(et):
        return False
    for a, b in zip(gt, et):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return False
        if abs(fa - fb) > 1e-6 * max(1.0, abs(fb)):
            return False
    return True


def out_key(out: str | None) -> str:
    """Canonical key of a program output (used for reference votes and execution signatures).

    Tokens are split on whitespace; a token containing ``.``, ``e`` or ``E`` that is not a plain integer and parses
    as a finite float is rewritten with ``%.6g`` (``-0.0`` becomes ``0``); tokens are joined by single spaces and the
    result is hashed to the first 16 hex digits of its SHA-1. ``None`` maps to the key of the empty output.
    """
    if out is None:
        out = ""
    res = []
    for t in out.split():
        if any(c in t for c in ".eE") and not t.lstrip("+-").isdigit():
            try:
                val = float(t)
            except ValueError:
                val = math.nan
            if math.isfinite(val):
                if val == 0.0:
                    val = 0.0
                res.append(f"{val:.6g}")
                continue
        res.append(t)
    return hashlib.sha1(" ".join(res).encode("utf-8", "replace")).hexdigest()[:16]

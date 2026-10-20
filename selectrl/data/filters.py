"""Keep problems whose answer is unique and that have enough tests.

Ported from SSPO v1 (data_pipeline.py is_unique_answer_problem / filter_problems).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from selectrl.sandbox.compare import out_key

MIN_PUBLIC = 1
MIN_HIDDEN = 5

MULTI_ANSWER_REGEX = re.compile(
    r"any of them|if there are (several|multiple)|print any|output any|return any|"
    r"any (valid|possible|such|order)|multiple (valid )?(answers|solutions)|"
    r"interactive problem|flush the output|sys\.stdout\.flush|"
    r"任意一个|输出任意|任一合法|多种答案|如果有多种|交互题",
    re.IGNORECASE,
)


def filter_reason(prob: dict[str, Any]) -> str:
    """``"kept"`` or the first reason to drop the problem.

    Reasons: empty_statement, multi_answer_statement (several accepted answers or interactive), insufficient_tests
    (< 1 public or < 5 hidden tests, or an empty expected output), inconsistent_expected_outputs (identical inputs with
    different canonical outputs).
    """
    question = str(prob.get("question") or "")
    if not question.strip():
        return "empty_statement"
    if MULTI_ANSWER_REGEX.search(question):
        return "multi_answer_statement"
    public, hidden = prob.get("public") or [], prob.get("hidden") or []
    if len(public) < MIN_PUBLIC or len(hidden) < MIN_HIDDEN:
        return "insufficient_tests"
    seen: dict[str, str] = {}
    for inp, exp in list(public) + list(hidden):
        if inp is None or exp is None or not str(exp).strip():
            return "insufficient_tests"
        k_in, k_out = str(inp).strip(), out_key(str(exp))
        if seen.setdefault(k_in, k_out) != k_out:
            return "inconsistent_expected_outputs"
    return "kept"


def filter_problems(problems: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Kept problems and statistics ``{total_raw, kept, removed_total, removed_fraction, removed_by_reason,
    removed_pids}``."""
    kept: list[dict[str, Any]] = []
    stats = FilterStats()
    for prob in problems:
        if stats.add(prob):
            kept.append(prob)
    return kept, stats.as_dict()


class FilterStats:
    """Streaming filter statistics (``add`` returns whether the problem is kept)."""

    def __init__(self) -> None:
        self.total = 0
        self.reasons: Counter[str] = Counter()
        self.removed: dict[str, list[str]] = {}

    def add(self, prob: dict[str, Any], reason: str | None = None) -> bool:
        """Record one problem with ``reason`` (default: ``filter_reason(prob)``); True if kept."""
        reason = reason or filter_reason(prob)
        self.total += 1
        if reason == "kept":
            return True
        self.reasons[reason] += 1
        self.removed.setdefault(reason, []).append(str(prob.get("pid")))
        return False

    def as_dict(self) -> dict[str, Any]:
        removed = sum(self.reasons.values())
        return {
            "total_raw": self.total,
            "kept": self.total - removed,
            "removed_total": removed,
            "removed_fraction": round(removed / max(1, self.total), 4),
            "removed_by_reason": dict(sorted(self.reasons.items())),
            "removed_pids": {k: sorted(v) for k, v in sorted(self.removed.items())},
        }

"""Zero reward for Mini-o3 validation smoke runs."""

from __future__ import annotations

import re
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
GROUNDING_RE = re.compile(r"<grounding>(.*?)</grounding>", re.DOTALL)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, float]:
    """Return neutral scores while preserving simple rollout diagnostics."""
    del data_source, ground_truth, extra_info, kwargs
    text = solution_str or ""
    has_answer = bool(ANSWER_RE.findall(text))
    grounding_open = text.count("<grounding>")
    grounding_close = text.count("</grounding>")
    think_balanced = text.count("<think>") == text.count("</think>")
    return {
        "score": 0.0,
        "acc": 0.0,
        "format_score": float(has_answer and grounding_open == grounding_close and think_balanced),
        "tool_call_count": float(len(GROUNDING_RE.findall(text))),
    }

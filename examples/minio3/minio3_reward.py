"""Mini-o3 rule reward for official verl custom_reward_function."""

from __future__ import annotations

import re
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
GROUNDING_RE = re.compile(r"<grounding>(.*?)</grounding>", re.DOTALL)


def _extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip()


def _normalize(text: Any) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(".")


def _matches_choice_or_exact(prediction: str, ground_truth: Any) -> float:
    pred = _normalize(prediction)
    gt = _normalize(ground_truth)
    if pred == gt:
        return 1.0

    choice = re.match(r"^\(([A-Z])\).*$", pred, re.DOTALL)
    if choice and choice.group(1) == gt:
        return 1.0

    choice = re.match(r"^([A-Z])\..*$", pred, re.DOTALL)
    if choice and choice.group(1) == gt:
        return 1.0

    return 0.0


def _format_score(response: str) -> float:
    has_answer = _extract_answer(response) is not None
    grounding_open = response.count("<grounding>")
    grounding_close = response.count("</grounding>")
    think_balanced = response.count("<think>") == response.count("</think>")
    return float(has_answer and grounding_open == grounding_close and think_balanced)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, float]:
    extra_info = extra_info or {}
    prediction = _extract_answer(solution_str)
    acc = 0.0 if prediction is None else _matches_choice_or_exact(prediction, ground_truth)
    fmt = _format_score(solution_str)
    tool_call_count = len(GROUNDING_RE.findall(solution_str or ""))

    acc_reward_weight = float(extra_info.get("acc_reward_weight", 1.0))
    format_reward_weight = float(extra_info.get("format_reward_weight", 0.0))
    tool_call_penalty = float(extra_info.get("tool_call_penalty", 0.0))
    use_tool_reward_weight = float(extra_info.get("use_tool_reward_weight", 0.0))

    tool_penalty_factor = (1.0 - tool_call_penalty) if tool_call_count > 0 else 1.0
    tool_reward = use_tool_reward_weight if tool_call_count > 0 else 0.0
    score = tool_penalty_factor * acc_reward_weight * acc + format_reward_weight * fmt + tool_reward

    return {
        "score": float(score),
        "acc": float(acc),
        "format_score": float(fmt),
        "tool_call_count": float(tool_call_count),
    }

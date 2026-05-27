"""Mini-o3 reward for official verl custom_reward_function."""

from __future__ import annotations

import os
import re
import time
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
GROUNDING_RE = re.compile(r"<grounding>(.*?)</grounding>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_MARKER_RE = re.compile(r"(?:final\s+answer|answer|答案)\s*[:：]\s*", re.IGNORECASE)
TEXT_UNIT_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*|[^\w\s]", re.UNICODE)
SYSTEM_PROMPT = "Judge answer equivalence. Reply only Yes or No."
QUERY_PROMPT = """Q: {question}
GT: {ground_truth}
Pred: {prediction}
Equivalent? Answer Yes or No."""
PLAIN_ANSWER_MIN_UNITS = 20
PLAIN_ANSWER_MAX_UNITS = 256
PLAIN_ANSWER_GT_UNIT_MULTIPLIER = 2


class JudgeConfigError(RuntimeError):
    """Raised when the requested LLM judge backend is not configured."""


_TEXT_JUDGE_CLIENT: Any | None = None
_TEXT_JUDGE_CLIENT_KEY: tuple[Any, ...] | None = None


def _extract_tagged_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip()


def _text_units(text: Any) -> list[re.Match[str]]:
    return list(TEXT_UNIT_RE.finditer(str(text or "")))


def _count_text_units(text: Any) -> int:
    return len(_text_units(text))


def _last_n_text_units(text: str, n_units: int) -> str:
    text = str(text or "")
    n_units = max(int(n_units), 0)
    if not text or n_units == 0:
        return ""

    units = _text_units(text)
    if len(units) <= n_units:
        return text
    return text[units[-n_units].start() :].strip()


def _plain_answer_unit_cap(ground_truth: Any) -> int:
    gt_units = _count_text_units(ground_truth)
    return max(
        PLAIN_ANSWER_MIN_UNITS,
        min(PLAIN_ANSWER_MAX_UNITS, PLAIN_ANSWER_GT_UNIT_MULTIPLIER * max(gt_units, 1)),
    )


def _strip_plain_final_text(text: str) -> str | None:
    text = str(text or "")
    if not text.strip():
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<tool_response>.*?</tool_response>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?</tool_call>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _last_sentence_or_answer_marker(text: str) -> str:
    marker_matches = list(ANSWER_MARKER_RE.finditer(text))
    if marker_matches:
        text = text[marker_matches[-1].end() :].strip()
    complete_sentences = re.findall(r"[^.!?。！？]+[.!?。！？]+(?:[\"'”’)\]]+)?", text)
    if complete_sentences:
        return complete_sentences[-1].strip()
    return text.strip()


def _extract_plain_final_answer(text: str, ground_truth: Any) -> str | None:
    text = _strip_plain_final_text(text)
    if text is None:
        return None
    text = _last_sentence_or_answer_marker(text)
    text = _last_n_text_units(text, _plain_answer_unit_cap(ground_truth))
    return text or None


def _extract_answer(text: str, *, relaxed: bool = False, ground_truth: Any = None) -> tuple[str | None, str]:
    tagged = _extract_tagged_answer(text)
    if tagged is not None:
        return tagged, "answer_tag"
    if relaxed:
        plain = _extract_plain_final_answer(text, ground_truth)
        if plain is not None:
            return plain, "plain_final"
    return None, "missing"


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
    has_answer = _extract_tagged_answer(response) is not None
    grounding_open = response.count("<grounding>")
    grounding_close = response.count("</grounding>")
    tool_call_open = response.count("<tool_call>")
    tool_call_close = response.count("</tool_call>")
    think_balanced = response.count("<think>") == response.count("</think>")
    return float(has_answer and grounding_open == grounding_close and tool_call_open == tool_call_close and think_balanced)


def _tool_call_count(response: str) -> int:
    return len(GROUNDING_RE.findall(response or "")) + len(TOOL_CALL_RE.findall(response or ""))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _retry_temperature(base_temperature: Any, attempt: int) -> float:
    base = 0.0 if base_temperature is None else float(base_temperature)
    return max(base, min(0.2 * max(int(attempt), 0), 1.0))


def _retry_sleep(attempt: int, initial_delay: Any) -> None:
    delay = max(float(initial_delay), 0.0) * (2 ** max(attempt - 1, 0))
    if delay > 0:
        time.sleep(min(delay, 30.0))


def _parse_judge_response(response_text: str) -> int:
    response_text = str(response_text or "")
    first_token = response_text.strip().split()[0].strip(".,:;!?'\"`").lower() if response_text.strip() else ""
    if first_token in {"yes", "1"}:
        return 1
    if first_token in {"no", "0"}:
        return 0

    match = re.search(r"\bscore\s*:\s*([01])\b", response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r'"score"\s*:\s*"?([01])"?', response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    raise ValueError(f"No judge yes/no or score found in response: {response_text}")


def _get_text_judge_client(
    provider: str,
    model: str | None,
    base_url: str | None,
    reasoning_effort: str | None,
):
    global _TEXT_JUDGE_CLIENT, _TEXT_JUDGE_CLIENT_KEY

    provider = (provider or "deepseek").lower()
    reasoning_effort = reasoning_effort or "none"
    if provider == "deepseek":
        model = model or os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash")
        base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise JudgeConfigError("DEEPSEEK_API_KEY is required when self_judge_provider=deepseek")
        thinking_type = "disabled" if reasoning_effort in {"", "none", None} else "enabled"
        cache_key = (provider, model, base_url, thinking_type)
        if _TEXT_JUDGE_CLIENT is None or _TEXT_JUDGE_CLIENT_KEY != cache_key:
            import openai

            _TEXT_JUDGE_CLIENT = openai.OpenAI(api_key=api_key, base_url=base_url)
            _TEXT_JUDGE_CLIENT_KEY = cache_key
            _TEXT_JUDGE_CLIENT._minio3_extra_body = {"thinking": {"type": thinking_type}}
        return _TEXT_JUDGE_CLIENT

    if provider == "openrouter":
        model = model or os.environ.get("OPENROUTER_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise JudgeConfigError("OPENROUTER_API_KEY is required when self_judge_provider=openrouter")
        cache_key = (provider, model, base_url, reasoning_effort)
        if _TEXT_JUDGE_CLIENT is None or _TEXT_JUDGE_CLIENT_KEY != cache_key:
            import openai

            _TEXT_JUDGE_CLIENT = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={
                    "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
                    "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Mini-o3 self judge"),
                },
            )
            _TEXT_JUDGE_CLIENT_KEY = cache_key
            _TEXT_JUDGE_CLIENT._minio3_extra_body = {
                "reasoning": {"effort": reasoning_effort, "exclude": True}
            }
        return _TEXT_JUDGE_CLIENT

    raise JudgeConfigError(f"Unsupported self_judge_provider={provider!r}")


def _query_llm_judge(
    *,
    prompt: str,
    provider: str,
    model: str | None,
    base_url: str | None,
    reasoning_effort: str | None,
    max_tokens: Any,
    temperature: Any,
    max_retries: Any,
    timeout: Any,
    initial_delay: Any,
) -> tuple[int, str, int]:
    client = _get_text_judge_client(provider, model, base_url, reasoning_effort)
    model = model or (
        os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash")
        if provider == "deepseek"
        else os.environ.get("OPENROUTER_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    )
    retries = max(int(max_retries), 1)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=_retry_temperature(temperature, attempt),
                max_tokens=int(max_tokens),
                extra_body=getattr(client, "_minio3_extra_body", None),
                timeout=float(timeout),
            )
            response_text = str(response.choices[0].message.content or "").strip()
            return _parse_judge_response(response_text), response_text, attempt + 1
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                _retry_sleep(attempt + 1, initial_delay)
    raise RuntimeError(f"LLM judge failed after {retries} attempts: {last_error}") from last_error


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    self_judge_reward: Any = False,
    self_judge_provider: str | None = None,
    self_judge_model: str | None = None,
    self_judge_base_url: str | None = None,
    self_judge_reasoning_effort: str | None = "none",
    self_judge_max_tokens: Any = 8,
    self_judge_temperature: Any = 0.0,
    self_judge_max_retries: Any | None = None,
    self_judge_timeout: Any = 60,
    self_judge_initial_delay: Any = 1.0,
    self_judge_relaxed_answer: Any = False,
    **kwargs,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    relaxed_answer = _truthy(self_judge_relaxed_answer) or _truthy(
        os.environ.get("MINIO3_RELAXED_ANSWER_EXTRACTION", False)
    )
    prediction, prediction_source = _extract_answer(
        solution_str,
        relaxed=relaxed_answer,
        ground_truth=ground_truth,
    )
    rule_acc = 0.0 if prediction is None else _matches_choice_or_exact(prediction, ground_truth)
    fmt = _format_score(solution_str)
    tool_call_count = _tool_call_count(solution_str or "")

    acc_reward_weight = float(extra_info.get("acc_reward_weight", 1.0))
    format_reward_weight = float(extra_info.get("format_reward_weight", 0.0))
    tool_call_penalty = float(extra_info.get("tool_call_penalty", 0.0))
    use_tool_reward_weight = float(extra_info.get("use_tool_reward_weight", 0.0))

    acc = rule_acc
    judge_fields: dict[str, Any] = {"rule_acc": float(rule_acc)}
    if _truthy(self_judge_reward):
        judge_fields.update(
            {
                "judge_score": 0.0,
                "judge_attempts": 0.0,
                "judge_source": "",
                "judge_response": "",
                "judge_prompt": "",
                "judge_error": "",
            }
        )
        provider = (self_judge_provider or os.environ.get("SELF_JUDGE_PROVIDER") or "deepseek").lower()
        if prediction is None or not str(prediction).strip():
            acc = 0.0
            judge_fields.update(
                {
                    "judge_score": 0.0,
                    "judge_attempts": 0.0,
                    "judge_source": "empty_answer",
                    "judge_response": "",
                    "judge_prompt": "",
                }
            )
        else:
            question = extra_info.get("question") or kwargs.get("prompt") or data_source
            judge_prompt = QUERY_PROMPT.format(question=question, ground_truth=ground_truth, prediction=prediction)
            retries = self_judge_max_retries
            if retries is None:
                retries = os.environ.get("LLM_JUDGE_MAX_RETRIES") or os.environ.get("SELF_JUDGE_MAX_RETRIES") or 5
            try:
                judge_score, judge_response, judge_attempts = _query_llm_judge(
                    prompt=judge_prompt,
                    provider=provider,
                    model=self_judge_model,
                    base_url=self_judge_base_url,
                    reasoning_effort=self_judge_reasoning_effort,
                    max_tokens=self_judge_max_tokens,
                    temperature=self_judge_temperature,
                    max_retries=retries,
                    timeout=self_judge_timeout,
                    initial_delay=self_judge_initial_delay,
                )
                acc = float(judge_score)
                judge_fields.update(
                    {
                        "judge_score": float(judge_score),
                        "judge_attempts": float(judge_attempts),
                        "judge_source": provider,
                        "judge_response": judge_response,
                        "judge_prompt": judge_prompt,
                    }
                )
            except JudgeConfigError:
                raise
            except Exception as exc:
                acc = 0.0
                judge_fields.update(
                    {
                        "judge_score": 0.0,
                        "judge_attempts": float(retries),
                        "judge_source": provider,
                        "judge_response": "",
                        "judge_prompt": judge_prompt,
                        "judge_error": str(exc),
                    }
                )

    tool_penalty_factor = (1.0 - tool_call_penalty) if tool_call_count > 0 else 1.0
    tool_reward = use_tool_reward_weight if tool_call_count > 0 else 0.0
    score = tool_penalty_factor * acc_reward_weight * acc + format_reward_weight * fmt + tool_reward

    result = {
        "score": float(score),
        "acc": float(acc),
        "format_score": float(fmt),
        "tool_call_count": float(tool_call_count),
        "answer_tag_present": float(_extract_tagged_answer(solution_str) is not None),
        "prediction_source": prediction_source,
        "prediction": prediction or "",
    }
    result.update(judge_fields)
    return result

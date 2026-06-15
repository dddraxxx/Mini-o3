# Copyright 2026 Mini-o3 contributors
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FINAL_ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:[*_`]+\s*)?(?:final\s+answer|答案)\s*[:：]\s*(?:[*_`]+\s*)?\S.*\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _stage_log(message: str) -> None:
    if os.getenv("MINIO3_STAGE_LOG", "0") == "1":
        logger.warning("[minio3-stage] %s", message)


def _has_terminal_final_answer(decoded: str) -> bool:
    decoded = str(decoded or "").strip()
    if not decoded:
        return False
    return FINAL_ANSWER_MARKER_RE.search(decoded) is not None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@register("mini_o3_tool_agent")
class MiniO3ToolAgentLoop(ToolAgentLoop):
    """Mini-o3 legacy grounding loop on top of official verl ToolAgentLoop.

    The model emits ``<grounding>{...}</grounding>``. We stop generation at the
    closing tag, crop the requested image region, and feed the crop back as a
    user observation so the next turn matches Mini-o3's training semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_budget = _as_bool(getattr(self.rollout_config.multi_turn, "show_budget", False))

    def _legacy_grounding_mode(self) -> bool:
        return self.tool_parser_name == "minio3_grounding"

    def _active_tool_schemas(self, agent_data: AgentData) -> list[dict[str, Any]]:
        return getattr(agent_data, "_active_tool_schemas", self.tool_schemas)

    def _initial_budget_text(self) -> str | None:
        if not self.show_budget or not self.max_assistant_turns:
            return None
        tool_rounds = max(int(self.max_assistant_turns) - 1, 0)
        return f"Budget: up to {tool_rounds} tool rounds."

    def _remaining_budget_text(self, agent_data: AgentData) -> str | None:
        if not self.show_budget:
            return None

        parts: list[str] = []
        if self.max_assistant_turns:
            tool_rounds_left = max(int(self.max_assistant_turns) - int(agent_data.assistant_turns) - 1, 0)
            parts.append(f"{tool_rounds_left} tool rounds left")

        remaining_tokens = max(int(self.response_length) - len(agent_data.response_mask), 0)
        parts.append(f"{self._format_token_budget(remaining_tokens)} tokens left")
        return f"Budget: {', '.join(parts)}."

    @staticmethod
    def _format_token_budget(tokens: int) -> str:
        if tokens >= 1000:
            return f"~{max(round(tokens / 1000), 1)}k"
        return str(max(tokens, 0))

    @classmethod
    def _append_text_to_message(cls, message: dict[str, Any], text: str) -> dict[str, Any]:
        updated = dict(message)
        content = updated.get("content")
        if isinstance(content, list):
            copied_content = [dict(item) if isinstance(item, dict) else item for item in content]
            if copied_content and isinstance(copied_content[-1], dict) and copied_content[-1].get("type") == "text":
                last = dict(copied_content[-1])
                last_text = str(last.get("text") or "").rstrip()
                last["text"] = f"{last_text}\n\n{text}" if last_text else text
                copied_content[-1] = last
            else:
                copied_content.append({"type": "text", "text": text})
            updated["content"] = copied_content
        else:
            existing = str(content or "").rstrip()
            updated["content"] = f"{existing}\n\n{text}" if existing else text
        return updated

    def _messages_with_initial_budget(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        budget_text = self._initial_budget_text()
        if not budget_text:
            return messages
        updated = list(messages)
        for idx in range(len(updated) - 1, -1, -1):
            if updated[idx].get("role") == "user":
                updated[idx] = self._append_text_to_message(updated[idx], budget_text)
                return updated
        return messages

    def _append_budget_to_text(self, text: str | None, agent_data: AgentData) -> str:
        budget_text = self._remaining_budget_text(agent_data)
        base = str(text or "").rstrip()
        if not budget_text:
            return base
        return f"{base}\n\n{budget_text}" if base else budget_text

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        turn_sampling_params = dict(sampling_params)
        if self._legacy_grounding_mode():
            turn_sampling_params.setdefault("stop", ["</grounding>"])
            turn_sampling_params.setdefault("include_stop_str_in_output", True)

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            self._mark_clip(agent_data, "response_length_before_generation")
            return AgentState.TERMINATED

        turn_sampling_params = dict(turn_sampling_params)
        remaining_tokens = max(self.response_length - len(agent_data.response_mask), 0)
        requested_max_tokens = turn_sampling_params.pop("max_tokens", turn_sampling_params.pop("max_new_tokens", None))
        if requested_max_tokens is None:
            turn_sampling_params["max_tokens"] = remaining_tokens
        else:
            turn_sampling_params["max_tokens"] = min(int(requested_max_tokens), remaining_tokens)

        t0 = time.monotonic()
        _stage_log(
            f"minio3.generate.start request={agent_data.request_id[:8]} "
            f"assistant_turn={agent_data.assistant_turns + 1} user_turns={agent_data.user_turns} "
            f"prompt_len={len(agent_data.prompt_ids)} remaining={remaining_tokens} "
            f"max_tokens={turn_sampling_params['max_tokens']}"
        )
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        _stage_log(
            f"minio3.generate.end request={agent_data.request_id[:8]} tokens={len(output.token_ids)} "
            f"stop={output.stop_reason} dt={time.monotonic() - t0:.3f}s"
        )

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            self._mark_clip(agent_data, "response_length")
            if agent_data.tool_calls:
                self._mark_exceed(agent_data, "response_length_with_tool_call")
            else:
                self._mark_format_if_missing_final(agent_data)
            return AgentState.TERMINATED

        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            if agent_data.tool_calls:
                self._mark_exceed(agent_data, "assistant_turn_limit_with_tool_call")
            else:
                self._mark_format_if_missing_final(agent_data)
            return AgentState.TERMINATED

        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            if agent_data.tool_calls:
                self._mark_exceed(agent_data, "user_turn_limit_with_tool_call")
            else:
                self._mark_format_if_missing_final(agent_data)
            return AgentState.TERMINATED

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS

        self._mark_format_if_missing_final(agent_data)
        return AgentState.TERMINATED

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        t0 = time.monotonic()
        _stage_log(
            f"minio3.pending.start request={agent_data.request_id[:8]} messages={len(agent_data.messages)} "
            f"images={len(agent_data.image_data or [])}"
        )
        schemas = None if self._legacy_grounding_mode() else self._active_tool_schemas(agent_data)
        prompt_ids = await self.apply_chat_template(
            self._messages_with_initial_budget(agent_data.messages),
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
        agent_data.prompt_ids = prompt_ids
        _stage_log(
            f"minio3.pending.end request={agent_data.request_id[:8]} prompt_len={len(prompt_ids)} "
            f"dt={time.monotonic() - t0:.3f}s"
        )
        return AgentState.GENERATING

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        if not self._legacy_grounding_mode():
            return await self._handle_official_processing_tools_state(agent_data)

        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        t0 = time.monotonic()
        _stage_log(f"minio3.tool.start request={agent_data.request_id[:8]} calls={len(agent_data.tool_calls)}")
        executed_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        tasks = []
        for tool_call in executed_tool_calls:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        for tool_call, (tool_response, tool_reward, tool_metrics) in zip(executed_tool_calls, responses, strict=True):
            self._record_tool_interaction(agent_data, tool_call, tool_response, tool_reward, tool_metrics)
            if tool_response.image:
                observation_id = len(agent_data.image_data or []) + len(new_images_this_turn)
                text = self._build_legacy_grounding_observation_text(agent_data.assistant_turns, observation_id)
                text = self._append_budget_to_text(text, agent_data)
                add_messages.append({"role": "user", "content": [{"type": "text", "text": text}, {"type": "image"}]})
                new_images_this_turn.extend([img for img in tool_response.image if img is not None])
            else:
                error_text = self._append_budget_to_text(
                    tool_response.text or "ERROR occurs during grounding.",
                    agent_data,
                )
                add_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{error_text}\n"
                            "Please analyze the error information and continue reasoning inside "
                            "<think> and </think>."
                        ),
                    }
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        response_ids = await self.apply_chat_template(
            add_messages,
            images=new_images_this_turn if new_images_this_turn else None,
            videos=None,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            self._mark_exceed(agent_data, "observation_would_exceed_response_length")
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)
            self._extend_raw_image_data(agent_data, new_images_this_turn)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        _stage_log(
            f"minio3.tool.end request={agent_data.request_id[:8]} new_images={len(new_images_this_turn)} "
            f"obs_tokens={len(response_ids)} dt={time.monotonic() - t0:.3f}s"
        )
        return AgentState.GENERATING

    async def _handle_official_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        t0 = time.monotonic()
        _stage_log(f"minio3.official_tool.start request={agent_data.request_id[:8]} calls={len(agent_data.tool_calls)}")
        executed_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        tasks = []
        tool_call_names = []
        for tool_call in executed_tool_calls:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        for tool_call, (tool_response, tool_reward, tool_metrics) in zip(executed_tool_calls, responses, strict=True):
            self._record_tool_interaction(agent_data, tool_call, tool_response, tool_reward, tool_metrics)
            visible_text = self._append_budget_to_text(tool_response.text, agent_data)
            if tool_response.image:
                content: list[dict[str, Any]] = [{"type": "image"}]
                if visible_text:
                    content.append({"type": "text", "text": visible_text})
                add_messages.append({"role": "tool", "content": content})
                new_images_this_turn.extend([img for img in tool_response.image if img is not None])
            else:
                error_text = self._append_budget_to_text(
                    tool_response.text or "ERROR occurs during tool execution.",
                    agent_data,
                )
                add_messages.append({"role": "tool", "content": error_text})

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        response_ids = await self._encode_tool_response_messages(
            add_messages,
            new_images_this_turn,
            tool_call_names=tool_call_names,
            image_start_index=self._count_media_items(agent_data.image_data) + 1,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            self._mark_exceed(agent_data, "official_tool_response_would_exceed_response_length")
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)
            self._extend_raw_image_data(agent_data, new_images_this_turn)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        _stage_log(
            f"minio3.official_tool.end request={agent_data.request_id[:8]} "
            f"new_images={len(new_images_this_turn)} obs_tokens={len(response_ids)} "
            f"dt={time.monotonic() - t0:.3f}s"
        )
        return AgentState.GENERATING

    @staticmethod
    def _build_legacy_grounding_observation_text(action_turn: int, observation_id: int) -> str:
        return (
            f"After the above Action {action_turn}, here is the zoom-in image "
            f"(Observation {observation_id}). Continue your reasoning process inside "
            "<think> and </think>. If needed, continue to zoom in on the original image "
            "or any observation by outputting <grounding> and </grounding> as before. "
            "If the final answer is confirmed, end with exactly one sentence: "
            "Final answer: <short answer>."
        )

    def _mark_exceed(self, agent_data: AgentData, reason: str) -> None:
        agent_data.extra_fields["exceed_mask"] = True
        agent_data.extra_fields.setdefault("exceed_reason", reason)
        self._mark_invalid(agent_data, f"exceed:{reason}")
        _stage_log(f"minio3.exceed request={agent_data.request_id[:8]} reason={reason}")

    def _mark_clip(self, agent_data: AgentData, reason: str) -> None:
        agent_data.extra_fields["clip_mask"] = True
        agent_data.extra_fields.setdefault("clip_reason", reason)
        self._mark_invalid(agent_data, f"clip:{reason}")
        _stage_log(f"minio3.clip request={agent_data.request_id[:8]} reason={reason}")

    def _mark_format_if_missing_final(self, agent_data: AgentData) -> None:
        decoded = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        if not _has_terminal_final_answer(decoded):
            reason = "missing_final_answer"
            agent_data.extra_fields["format_mask"] = True
            agent_data.extra_fields.setdefault("format_reason", reason)
            self._mark_invalid(agent_data, f"format:{reason}")
            _stage_log(f"minio3.format request={agent_data.request_id[:8]} reason={reason}")

    def _mark_invalid(self, agent_data: AgentData, reason: str) -> None:
        agent_data.extra_fields["invalid_mask"] = True
        reasons = agent_data.extra_fields.get("invalid_reasons")
        if reasons is None:
            reasons = []
        elif not isinstance(reasons, list):
            reasons = [reasons]
        if reason not in reasons:
            reasons.append(reason)
        agent_data.extra_fields["invalid_reasons"] = reasons

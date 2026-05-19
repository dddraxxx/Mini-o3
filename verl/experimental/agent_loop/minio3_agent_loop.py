# Copyright 2026 Mini-o3 contributors
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("mini_o3_tool_agent")
class MiniO3ToolAgentLoop(ToolAgentLoop):
    """Mini-o3 legacy grounding loop on top of official verl ToolAgentLoop.

    The model emits ``<grounding>{...}</grounding>``. We stop generation at the
    closing tag, crop the requested image region, and feed the crop back as a
    user observation so the next turn matches Mini-o3's training semantics.
    """

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        turn_sampling_params = dict(sampling_params)
        if self.tool_parser_name == "minio3_grounding":
            turn_sampling_params.setdefault("stop", ["</grounding>"])
            turn_sampling_params.setdefault("include_stop_str_in_output", True)
        return await super()._handle_generating_state(agent_data, turn_sampling_params, ignore_termination)

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=None,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        for tool_response, tool_reward, _ in responses:
            if tool_response.image:
                observation_id = len(agent_data.image_data or []) + len(new_images_this_turn)
                text = self._build_observation_text(agent_data.assistant_turns, observation_id)
                add_messages.append({"role": "user", "content": [{"type": "text", "text": text}, {"type": "image"}]})
                new_images_this_turn.extend([img for img in tool_response.image if img is not None])
            else:
                error_text = tool_response.text or "ERROR occurs during grounding."
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
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    @staticmethod
    def _build_observation_text(action_turn: int, observation_id: int) -> str:
        return (
            f"After the above Action {action_turn}, here is the zoom-in image "
            f"(Observation {observation_id}). Continue your reasoning process inside "
            "<think> and </think>. If needed, continue to zoom in on the original image "
            "or any observation by outputting <grounding> and </grounding> as before. "
            "If the final answer is confirmed, put it inside <answer> and </answer>."
        )

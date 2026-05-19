import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _align_non_tensor_batch_keys, prompt_admission_reward_stats


class _FakePromptAdmissionManager:
    def __init__(self, reward_groups: list[torch.Tensor]):
        self.reward_groups = list(reward_groups)
        self.submitted = 0
        self.collected = 0
        self.cancelled = 0
        self.submitted_prompt_labels = []

    def submit_prompt_group(self, rollout_prompt: DataProto, group_id: str) -> dict:
        raw_prompt = rollout_prompt.non_tensor_batch.get("raw_prompt")
        prompt_label = str(raw_prompt[0]) if raw_prompt is not None else group_id
        self.submitted_prompt_labels.append(prompt_label)
        handle = {
            "group_id": group_id,
            "prompt_len": len(rollout_prompt),
            "prompt_label": prompt_label,
            "worker_idx": self.submitted % 2,
        }
        self.submitted += 1
        return handle

    def wait_prompt_groups(self, running: list[dict], timeout: float | None = None) -> list[dict]:
        return running[:1]

    def collect_prompt_group(self, handle: dict) -> DataProto:
        reward_tensor = self.reward_groups[self.collected]
        self.collected += 1
        return DataProto.from_single_dict(
            {
                "responses": torch.full((reward_tensor.shape[0], 2), fill_value=self.collected, dtype=torch.long),
                "response_mask": torch.ones(reward_tensor.shape[0], 2, dtype=torch.long),
                "rm_scores": reward_tensor,
                "source_prompt": np.array([handle["prompt_label"]] * reward_tensor.shape[0], dtype=object),
            },
            meta_info={"timing": {"fake_rollout": 0.25}},
        )

    def cancel_prompt_group(self, handle: dict) -> bool:
        self.cancelled += 1
        return True

    def prompt_admission_status(self) -> dict:
        return {
            "manager_inflight": 0,
            "worker_inflight": [0, 0],
            "submitted_total": self.submitted,
            "collected_total": self.collected,
        }


def _prompt_dict(prompt_id: int) -> dict:
    return {
        "input_ids": torch.tensor([[prompt_id, prompt_id + 1, prompt_id + 2]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "position_ids": torch.arange(3, dtype=torch.long).unsqueeze(0),
        "raw_prompt": np.array([f"prompt-{prompt_id}"], dtype=object),
    }


def _trainer_with_prompt_admission(
    tmp_path,
    reward_groups: list[torch.Tensor],
    *,
    pool_size: int = 1,
) -> RayPPOTrainer:
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"temperature": 1.0, "n": 2}},
            "algorithm": {"adv_estimator": "grpo", "max_num_gen_batches": 3},
            "data": {"train_batch_size": 1},
            "trainer": {
                "prompt_admission_pool_size": pool_size,
                "prompt_admission_reward_std_epsilon": 1.0e-4,
                "prompt_admission_wait_timeout_s": 0.0,
                "prompt_admission_cancel_unfinished": True,
                "prompt_admission_state_path": str(tmp_path / "prompt_admission_state.jsonl"),
            },
        }
    )
    trainer.global_steps = 7
    trainer.async_rollout_manager = _FakePromptAdmissionManager(reward_groups)
    return trainer


def test_prompt_admission_rejects_constant_reward_groups():
    reward_tensor = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    accepted, stats = prompt_admission_reward_stats(reward_tensor, epsilon=1e-4)

    assert accepted is False
    assert stats["reward_std"] == pytest.approx(0.0)
    assert stats["reward_min"] == pytest.approx(1.0)
    assert stats["reward_max"] == pytest.approx(1.0)


def test_prompt_admission_accepts_reward_variation_within_group():
    reward_tensor = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
        ]
    )

    accepted, stats = prompt_admission_reward_stats(reward_tensor, epsilon=1e-4)

    assert accepted is True
    assert stats["reward_std"] > 1e-4
    assert stats["reward_min"] == pytest.approx(0.0)
    assert stats["reward_max"] == pytest.approx(1.0)


def test_prompt_admission_rejects_single_rollout_groups():
    reward_tensor = torch.tensor([[0.0, 0.0, 1.0]])

    accepted, stats = prompt_admission_reward_stats(reward_tensor, epsilon=1e-4)

    assert accepted is False
    assert stats["reward_std"] == pytest.approx(0.0)


def test_prompt_admission_aligns_sparse_extra_fields_before_concat():
    first = DataProto.from_single_dict(
        {
            "responses": torch.ones(2, 2, dtype=torch.long),
            "source_prompt": np.array(["a", "a"], dtype=object),
        }
    )
    second = DataProto.from_single_dict(
        {
            "responses": torch.ones(1, 2, dtype=torch.long),
            "source_prompt": np.array(["b"], dtype=object),
            "exceed_reason": np.array(["length"], dtype=object),
        }
    )

    _align_non_tensor_batch_keys([first, second])
    merged = DataProto.concat([first, second])

    assert merged.non_tensor_batch["source_prompt"].tolist() == ["a", "a", "b"]
    assert merged.non_tensor_batch["exceed_reason"].tolist() == [None, None, "length"]


def test_prompt_admission_collects_until_batch_is_admitted(tmp_path):
    trainer = _trainer_with_prompt_admission(
        tmp_path,
        reward_groups=[
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        ],
    )
    metrics = {}

    admitted_batch, reward_tensor, reward_extra_infos = trainer._collect_prompt_admitted_batch(
        _prompt_dict(10),
        iter([_prompt_dict(20)]),
        metrics,
    )

    assert len(admitted_batch) == 2
    assert reward_tensor.sum(dim=-1).tolist() == [0.0, 1.0]
    assert reward_extra_infos == {}
    assert len(set(admitted_batch.non_tensor_batch["uid"].tolist())) == 1
    assert trainer.async_rollout_manager.submitted == 2
    assert trainer.async_rollout_manager.collected == 2
    assert metrics["prompt_admission/submitted_groups"] == pytest.approx(2.0)
    assert metrics["prompt_admission/rejected_groups"] == pytest.approx(1.0)
    assert metrics["prompt_admission/accepted_groups"] == pytest.approx(1.0)
    assert metrics["prompt_admission/fetched_batches"] == pytest.approx(1.0)
    assert metrics["prompt_admission/pending_groups"] == pytest.approx(0.0)
    assert metrics["prompt_admission/timing/fake_rollout"] == pytest.approx(0.5)

    state_rows = (tmp_path / "prompt_admission_state.jsonl").read_text().splitlines()
    assert state_rows
    state = json.loads(state_rows[-1])
    assert state["step"] == 7
    assert state["pending_group_count"] == 0
    assert state["metrics"]["prompt_admission/submitted_groups"] == pytest.approx(2.0)


def test_prompt_admission_reuses_unfinished_group_across_steps(tmp_path):
    trainer = _trainer_with_prompt_admission(
        tmp_path,
        reward_groups=[
            torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 2.0]]),
        ],
        pool_size=2,
    )
    first_metrics = {}

    trainer._collect_prompt_admitted_batch(
        _prompt_dict(10),
        iter([_prompt_dict(20)]),
        first_metrics,
    )

    assert first_metrics["prompt_admission/submitted_groups"] == pytest.approx(1.0)
    assert first_metrics.get("prompt_admission/unfinished_running_groups", 0.0) == pytest.approx(0.0)
    assert first_metrics.get("prompt_admission/cancelled_running_groups", 0.0) == pytest.approx(0.0)
    assert first_metrics["prompt_admission/pending_groups"] == pytest.approx(1.0)
    assert first_metrics["prompt_admission/reused_pending_groups"] == pytest.approx(0.0)

    trainer.config.trainer.prompt_admission_pool_size = 1
    second_metrics = {}
    second_batch, second_reward, _ = trainer._collect_prompt_admitted_batch(
        _prompt_dict(30),
        iter([]),
        second_metrics,
    )

    assert second_reward.sum(dim=-1).tolist() == [0.0, 2.0]
    assert second_batch.non_tensor_batch["source_prompt"].tolist() == ["prompt-20", "prompt-20"]
    assert trainer.async_rollout_manager.submitted_prompt_labels[:2] == ["prompt-10", "prompt-20"]
    assert second_metrics["prompt_admission/reused_pending_groups"] == pytest.approx(0.0)
    assert second_metrics["prompt_admission/pending_groups"] == pytest.approx(1.0)

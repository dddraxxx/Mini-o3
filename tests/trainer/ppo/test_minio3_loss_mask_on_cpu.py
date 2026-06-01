import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_minio3_loss_masks


def test_apply_minio3_loss_masks_zeroes_enabled_exceed_rows():
    data = DataProto.from_single_dict(
        {
            "response_mask": torch.ones(3, 4, dtype=torch.long),
            "responses": torch.ones(3, 4, dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0],
                ],
                dtype=torch.long,
            ),
            "exceed_mask": np.array([False, True, None], dtype=object),
            "format_mask": np.array([False, False, True], dtype=object),
        }
    )

    metrics = apply_minio3_loss_masks(data, {"ignore_exceed": True, "ignore_format": False})

    assert data.batch["response_mask"].tolist() == [
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ]
    assert metrics["batch/clip_sample_ratio"] == pytest.approx(1 / 3)
    assert metrics["batch/exceed_sample_ratio"] == pytest.approx(1 / 3)
    assert metrics["batch/format_sample_ratio"] == pytest.approx(1 / 3)
    assert metrics["batch/invalid_sample_ratio"] == pytest.approx(2 / 3)
    assert metrics["batch/minio3_loss_mask_zeroed_ratio"] == pytest.approx(1 / 3)


def test_apply_minio3_loss_masks_can_zero_invalid_rows():
    data = DataProto.from_single_dict(
        {
            "response_mask": torch.ones(2, 3, dtype=torch.long),
            "responses": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long),
            "exceed_mask": np.array([False, True], dtype=object),
            "format_mask": np.array([True, False], dtype=object),
        }
    )

    metrics = apply_minio3_loss_masks(data, {"ignore_invalid": True})

    assert data.batch["response_mask"].tolist() == [
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert metrics["batch/minio3_loss_mask_zeroed_ratio"] == 1.0
    assert metrics["batch/ignore_invalid_sample_ratio"] == 1.0

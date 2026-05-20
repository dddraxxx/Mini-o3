import pytest

from examples.minio3 import minio3_reward


def test_rule_reward_matches_choice_by_default():
    result = minio3_reward.compute_score(
        data_source="visual_probe_easy",
        solution_str="<think>x</think><answer>(A) red</answer>",
        ground_truth="A",
    )

    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["rule_acc"] == 1.0


def test_self_judge_short_circuits_empty_answer(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("judge should not be called for empty answers")

    monkeypatch.setattr(minio3_reward, "_query_llm_judge", fail_if_called)

    result = minio3_reward.compute_score(
        data_source="visual_probe_easy",
        solution_str="<think>x</think>",
        ground_truth="A",
        extra_info={"question": "What color is it?"},
        self_judge_reward=True,
    )

    assert result["score"] == 0.0
    assert result["acc"] == 0.0
    assert result["judge_score"] == 0.0
    assert result["judge_source"] == "empty_answer"
    assert result["judge_attempts"] == 0.0


def test_self_judge_overrides_rule_exact_match(monkeypatch):
    def fake_judge(**kwargs):
        assert "GT: flip-flops" in kwargs["prompt"]
        assert "Pred: sandals" in kwargs["prompt"]
        return 1, "Yes", 1

    monkeypatch.setattr(minio3_reward, "_query_llm_judge", fake_judge)

    result = minio3_reward.compute_score(
        data_source="visual_probe_easy",
        solution_str="<think>x</think><answer>sandals</answer>",
        ground_truth="flip-flops",
        extra_info={"question": "What footwear is shown?"},
        self_judge_reward=True,
        self_judge_provider="deepseek",
    )

    assert result["rule_acc"] == 0.0
    assert result["acc"] == 1.0
    assert result["score"] == 1.0
    assert result["judge_score"] == 1.0
    assert result["judge_response"] == "Yes"


def test_deepseek_judge_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(minio3_reward, "_TEXT_JUDGE_CLIENT", None)
    monkeypatch.setattr(minio3_reward, "_TEXT_JUDGE_CLIENT_KEY", None)

    with pytest.raises(minio3_reward.JudgeConfigError, match="DEEPSEEK_API_KEY"):
        minio3_reward.compute_score(
            data_source="visual_probe_easy",
            solution_str="<think>x</think><answer>A</answer>",
            ground_truth="A",
            extra_info={"question": "Pick one."},
            self_judge_reward=True,
            self_judge_provider="deepseek",
        )

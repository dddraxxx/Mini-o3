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


def test_self_judge_relaxed_answer_uses_plain_final_text(monkeypatch):
    def fake_judge(**kwargs):
        assert "Pred: Based on the visual evidence, the animal is a white stork." in kwargs["prompt"]
        return 1, "Yes", 1

    monkeypatch.setattr(minio3_reward, "_query_llm_judge", fake_judge)

    result = minio3_reward.compute_score(
        data_source="visual_probe_easy",
        solution_str="<think>I inspected the image.</think> Based on the visual evidence, the animal is a white stork.",
        ground_truth="a bird",
        extra_info={"question": "What animal is shown?"},
        self_judge_reward=True,
        self_judge_provider="deepseek",
        self_judge_relaxed_answer=True,
    )

    assert result["answer_tag_present"] == 0.0
    assert result["prediction_source"] == "plain_final"
    assert result["acc"] == 1.0
    assert result["score"] == 1.0


def test_self_judge_relaxed_answer_caps_plain_final_units(monkeypatch):
    captured = {}

    def fake_judge(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return 1, "Yes", 1

    monkeypatch.setattr(minio3_reward, "_query_llm_judge", fake_judge)

    repeated = " ".join(f"word{i}" for i in range(60))
    result = minio3_reward.compute_score(
        data_source="visual_probe_easy",
        solution_str=f"<think>I inspected the image.</think> {repeated}",
        ground_truth="blue",
        extra_info={"question": "What color is it?"},
        self_judge_reward=True,
        self_judge_provider="deepseek",
        self_judge_relaxed_answer=True,
    )

    assert result["prediction_source"] == "plain_final"
    assert result["prediction"] == " ".join(f"word{i}" for i in range(40, 60))
    assert f"Pred: {result['prediction']}" in captured["prompt"]
    assert "word39" not in captured["prompt"]


def test_self_judge_relaxed_answer_prefers_last_complete_sentence(monkeypatch):
    def fake_judge(**kwargs):
        assert "Pred: The person is wearing red clothes." in kwargs["prompt"]
        return 1, "Yes", 1

    monkeypatch.setattr(minio3_reward, "_query_llm_judge", fake_judge)

    result = minio3_reward.compute_score(
        data_source="visual_probe_easy",
        solution_str=(
            "<think>I inspected the image.</think> "
            "The person is wearing red clothes. "
            "The person is wearing red clothes. "
            "The person is"
        ),
        ground_truth="red clothes",
        extra_info={"question": "What is the person wearing?"},
        self_judge_reward=True,
        self_judge_provider="deepseek",
        self_judge_relaxed_answer=True,
    )

    assert result["prediction"] == "The person is wearing red clothes."


def test_self_judge_error_fields_are_stable(monkeypatch):
    def failing_judge(**kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(minio3_reward, "_query_llm_judge", failing_judge)

    result = minio3_reward.compute_score(
        data_source="visual_probe_easy",
        solution_str="<think>x</think><answer>blue</answer>",
        ground_truth="blue",
        extra_info={"question": "What color is it?"},
        self_judge_reward=True,
        self_judge_provider="deepseek",
        self_judge_max_retries=3,
    )

    assert result["score"] == 0.0
    assert result["judge_score"] == 0.0
    assert result["judge_attempts"] == 3.0
    assert result["judge_source"] == "deepseek"
    assert "quota exceeded" in result["judge_error"]


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

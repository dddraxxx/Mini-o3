# Qwen3.5-9B VisualProbe Full Evaluation

This records the full VisualProbe validation runs used as the current Qwen3.5-9B baseline and SFT comparisons for Mini-o3 with the official zoom tool surface.

## Dataset

- VisualProbe validation split, stratified as Easy, Medium, Hard.
- Total examples: 515
- Split sizes: Easy 141, Medium 268, Hard 106
- Images are read from `data/VisualProbe_Easy`, `data/VisualProbe_Medium`, and `data/VisualProbe_Hard`.
- Frozen preprocessing entrypoint: `exps/eval/snapshots/20260525_qwen35_vp/prepare_visualprobe_val_smoke.py`

## Shared Eval Settings

- Model: `Qwen/Qwen3.5-9B`
- Successful latest run used local snapshot:
  `/mnt/localssd/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Agent loop: `mini_o3_tool_agent`
- Tool: `image_zoom_in_tool`
- Multi-turn format: `qwen3_coder`
- Reward: `exps/eval/snapshots/20260525_qwen35_vp/minio3_reward.py`
- Judge: DeepSeek `deepseek-v4-flash`, relaxed answer matching
- Validation sampling: `VAL_N=1`, `VAL_DO_SAMPLE=True`, `VAL_TEMPERATURE=1.0`, `VAL_TOP_P=1.0`, `VAL_TOP_K=-1`
- Validation batch: `VAL_BATCH_SIZE=512`
- Prompt/response lengths: `MAX_PROMPT_LENGTH=16384`, `MAX_RESPONSE_LENGTH=16384`, `VAL_RESPONSE_LENGTH=32768`, `MAX_MODEL_LEN=65536`
- Rollout: `ROLLOUT_DP=8`, `ROLLOUT_TP=1`, `ROLLOUT_GPU_MEM_UTIL=0.9`, `MAX_NUM_BATCHED_TOKENS=65536`, `MAX_NUM_SEQS=256`
- Agent workers: `AGENT_NUM_WORKERS=64`
- Ray CPUs: `RAY_NUM_CPUS=96`
- Max turns: train `6/6`, val `12/12`

The eval is not bit-stable: model sampling uses temperature 1.0 and the final score depends on an external DeepSeek judge. Treat these as reproducible settings, not an exact deterministic seed.

## Results

| Run | Prompt suite | Overall | Easy | Medium | Hard | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `visualprobe_full515_qwen35_9b_official_tool_plainq_minio3agent_localpath_deepseek_relaxed_20260525_131313` | `qwen35_official_zoom_tool_plain_question` | 191/515 = 37.09% | 74/141 = 52.48% | 89/268 = 33.21% | 28/106 = 26.42% | Clean prompt: image then question, no answer tag instruction |
| `visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_plainq_deepseek_20260530_1251` | `qwen35_official_zoom_tool_plain_question` | 130/515 = 25.24% | 51/141 = 36.17% | 66/268 = 24.63% | 13/106 = 12.26% | Full-language SFT, frozen vision/projector, global batch 32 |
| `visualprobe_full515_qwen35_9b_official_tool_deepseek_relaxed_retry_20260525_024413` | `qwen35_official_zoom_tool` | 182/515 = 35.34% | 75/141 = 53.19% | 84/268 = 31.34% | 23/106 = 21.70% | Earlier answer-tag prompt suite |

## Plain-Question Run Details

- Data dir: `data/minio3_visualprobe_val_plain_question515_minio3agent_localpath`
- Log: `logs/visualprobe_full515_qwen35_9b_official_tool_plainq_minio3agent_localpath_deepseek_relaxed_20260525_131313.log`
- Run dir: `save/visualprobe_full515_qwen35_9b_official_tool_plainq_minio3agent_localpath_deepseek_relaxed_20260525_131313`
- Generations: `save/visualprobe_full515_qwen35_9b_official_tool_plainq_minio3agent_localpath_deepseek_relaxed_20260525_131313/validation_generations/0.jsonl`
- Metrics: `save/visualprobe_full515_qwen35_9b_official_tool_plainq_minio3agent_localpath_deepseek_relaxed_20260525_131313/train_step_metrics.jsonl`
- Exit: clean, `exit 0`

Additional stats:

- Prediction source: `plain_final=513`, `missing=2`
- Answer tag present: `0/515`
- DeepSeek judged: `513/515`
- Empty predictions: `2`
- Exceed reason: `assistant_turn_limit_with_tool_call=2`
- Tool call mean: Easy `2.525`, Medium `2.396`, Hard `3.302`
- Number of turns: min `2`, max `24`, mean `7.227`

## SFT Full-Freeze Run Details

- Model: `save/qwen35_9b_official_tool_h200_sft_full_freeze_20260530_101126/global_step_681_hf`
- Data dir: `data/minio3_visualprobe_val_plain_question515_minio3agent_localpath`
- Log: `logs/visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_plainq_deepseek_20260530_1251.log`
- Run dir: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_plainq_deepseek_20260530_1251`
- Generations: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_plainq_deepseek_20260530_1251/validation_generations/0.jsonl`
- Metrics: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_plainq_deepseek_20260530_1251/train_step_metrics.jsonl`
- Exit: clean, `exit 0`

Additional stats:

- Prediction source: mostly `plain_final`; answer tag is not requested in this prompt suite.
- Answer tag present: `0/515`
- Tool call mean: Easy `6.752`, Medium `6.563`, Hard `9.566`
- DeepSeek judge attempts mean: Easy `0.716`, Medium `0.761`, Hard `0.396`
- Number of turns: min `2`, max `24`, mean `15.810`
- Generation output length in `validation_generations/0.jsonl`: median about 10.4k characters, max about 120k characters.
- Comparison to the base plain-question run: lower score on all splits and substantially more tool calls/turns, indicating the global-batch-32 full SFT learned the tool-use trajectory style but became less efficient on VP.

## Answer-Tag Run Details

- Data dir: `data/minio3_visualprobe_val_smoke515`
- Log: `logs/visualprobe_full515_qwen35_9b_official_tool_deepseek_relaxed_retry_20260525_024413.log`
- Run dir: `save/visualprobe_full515_qwen35_9b_official_tool_deepseek_relaxed_retry_20260525_024413`
- Generations: `save/visualprobe_full515_qwen35_9b_official_tool_deepseek_relaxed_retry_20260525_024413/validation_generations/0.jsonl`
- Metrics: `save/visualprobe_full515_qwen35_9b_official_tool_deepseek_relaxed_retry_20260525_024413/train_step_metrics.jsonl`
- Exit: clean, `exit 0`

Additional stats:

- Prediction source: `plain_final=277`, `answer_tag=230`, `missing=8`
- Answer tag present: `230/515`
- DeepSeek judged: `507/515`
- Empty predictions: `8`
- Exceed reason: `assistant_turn_limit_with_tool_call=8`
- Tool call mean: Easy `2.411`, Medium `2.407`, Hard `3.340`

## Reproduction

Use:

```bash
bash exps/eval/run_visualprobe_qwen35_full_eval.sh plain_question
```

For the earlier answer-tag prompt:

```bash
bash exps/eval/run_visualprobe_qwen35_full_eval.sh answer_tag
```

The script launches a tmux session by default and writes logs under `logs/`, data under `data/`, and run outputs under `save/`. It reads `DEEPSEEK_API_KEY` and `HF_TOKEN` from the environment, falling back to `/mnt/localssd/AGENTS.md` when available. It does not print either token.

The top-level runner is intentionally thin and delegates to the frozen snapshot under `exps/eval/snapshots/20260525_qwen35_vp/`, not the live `examples/minio3` wrappers. The snapshot includes:

- `run_real_val_visualprobe_smoke.sh`
- `run_qwen3_vl_8b_crop_lora_fsdp.sh`
- `prepare_visualprobe_val_smoke.py`
- `preprocess_visualprobe.py`
- `check_qwen35_env.py`
- `monitor_gpu_util.py`
- `summarize_run_metrics.py`
- `minio3_reward.py`
- `empty_reward.py`
- `config/tool_config/minio3_image_zoom_in_tool.yaml`
- `config/tool_config/minio3_crop_tool.yaml`

This freezes the eval launch path and prompt/reward/tool configuration. It still runs the repo's installed `verl`/Mini-o3 Python package through `uv --project`, so it is a reproducible eval script snapshot rather than a full source-code archive.

Useful overrides:

```bash
MODEL_PATH=/path/to/Qwen3.5-9B/snapshot bash exps/eval/run_visualprobe_qwen35_full_eval.sh plain_question
MINIO3_EVAL_FOREGROUND=1 bash exps/eval/run_visualprobe_qwen35_full_eval.sh plain_question
RAY_NUM_CPUS=128 AGENT_NUM_WORKERS=128 bash exps/eval/run_visualprobe_qwen35_full_eval.sh plain_question
```

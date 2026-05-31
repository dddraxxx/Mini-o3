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
- Vision IDs: `ADD_VISION_ID=True`, so the initial image and tool observation images are labeled as `Picture N`.
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
| `visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_finalsentence_deepseek_20260530` | `qwen35_official_zoom_tool_final_sentence` | 139/515 = 26.99% | 57/141 = 40.43% | 66/268 = 24.63% | 16/106 = 15.09% | SFT/RL-aligned final-answer prompt with `ADD_VISION_ID=True` |
| `visualprobe_full515_qwen35_9b_sft_full_freeze_gbs128_tok32k_gs168_finalsentence_deepseek_ray64_20260530` | `qwen35_official_zoom_tool_final_sentence` | 144/515 = 27.96% | 60/141 = 42.55% | 67/268 = 25.00% | 17/106 = 16.04% | Full-language SFT, frozen vision/projector, global batch 128 |
| `visualprobe_full515_qwen35_9b_sft_full_freeze_gbs256_tok32k_gs84_finalsentence_deepseek_20260531_rerun1` | `qwen35_official_zoom_tool_final_sentence` | 129/515 = 25.05% | 58/141 = 41.13% | 58/268 = 21.64% | 13/106 = 12.26% | Full-language SFT, frozen vision/projector, global batch 256 |
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

## SFT Final-Sentence Run Details

- Model: `save/qwen35_9b_official_tool_h200_sft_full_freeze_20260530_101126/global_step_681_hf`
- Data dir: `data/minio3_visualprobe_val_final_sentence515_minio3agent_localpath`
- Log: `logs/visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_finalsentence_deepseek_20260530.log`
- Run dir: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_finalsentence_deepseek_20260530`
- Generations: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_finalsentence_deepseek_20260530/validation_generations/0.jsonl`
- Metrics: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gs681_finalsentence_deepseek_20260530/train_step_metrics.jsonl`
- Exit: clean, `exit 0`. A vLLM EngineCore shutdown message appeared after generations were dumped, but the wrapper exited cleanly and metrics/generations are complete.

Additional stats:

- `Final answer:` present: `433/515`
- `Picture` labels present: `515/515`
- Tool calls with `img_idx > 0`: `1486/3048`; rows using nonzero observation image index: `432/515`
- Empty predictions: `81`, all corresponding to `tool_call_count >= 12`
- DeepSeek judged: `434/515`
- Prediction length after final-answer extraction: mean `15` chars, p95 `75`, max `127`
- Tool call mean: Easy `5.567`, Medium `5.623`, Hard `7.906`
- Number of turns: min `2`, max `24`, mean `13.841`

## SFT GBS128 Final-Sentence Run Details

- Model: `save/qwen35_9b_official_tool_h200_sft_full_freeze_gbs128_tok32k_20260530_143908/global_step_168_hf`
- Data dir: `data/minio3_visualprobe_val_final_sentence515_minio3agent_localpath`
- Log: `logs/visualprobe_full515_qwen35_9b_sft_full_freeze_gbs128_tok32k_gs168_finalsentence_deepseek_ray64_20260530.log`
- Run dir: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gbs128_tok32k_gs168_finalsentence_deepseek_ray64_20260530`
- Generations: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gbs128_tok32k_gs168_finalsentence_deepseek_ray64_20260530/validation_generations/0.jsonl`
- Metrics: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gbs128_tok32k_gs168_finalsentence_deepseek_ray64_20260530/train_step_metrics.jsonl`
- Exit: clean, `exit 0`

Additional stats:

- `Final answer:` present: `430/515`
- `Picture` labels present: `515/515`
- Empty predictions: `83`, all corresponding to `tool_call_count >= 12`
- DeepSeek judge attempts mean: Easy `0.887`, Medium `0.866`, Hard `0.708`
- Prediction length after final-answer extraction: mean `16` chars, p95 `79`, max `118`
- Generation output length in `validation_generations/0.jsonl`: median `7121` chars, p95 `18135`, max `25800`
- Tool call mean: Easy `5.716`, Medium `5.534`, Hard `7.217`

## SFT GBS256 Final-Sentence Run Details

- Model: `save/qwen35_9b_official_tool_h200_sft_full_freeze_gbs256_tok32k_20260530_235204/global_step_84_hf`
- Data dir: `data/minio3_visualprobe_val_final_sentence515_minio3agent_localpath`
- Log: `logs/visualprobe_full515_qwen35_9b_sft_full_freeze_gbs256_tok32k_gs84_finalsentence_deepseek_20260531_rerun1.log`
- Run dir: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gbs256_tok32k_gs84_finalsentence_deepseek_20260531_rerun1`
- Generations: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gbs256_tok32k_gs84_finalsentence_deepseek_20260531_rerun1/validation_generations/0.jsonl`
- Metrics: `save/visualprobe_full515_qwen35_9b_sft_full_freeze_gbs256_tok32k_gs84_finalsentence_deepseek_20260531_rerun1/train_step_metrics.jsonl`
- Exit: clean, `exit 0`

Additional stats:

- `Final answer:` present: `411/515`
- `Picture` labels present: `515/515`
- Empty predictions: `103`, all corresponding to `tool_call_count >= 12`
- DeepSeek judge attempts mean: Easy `0.823`, Medium `0.832`, Hard `0.689`
- Prediction length after final-answer extraction: mean `15` chars, p95 `71`, max `106`
- Generation output length in `validation_generations/0.jsonl`: median `7922` chars, p95 `19280`, max `24919`
- Tool call mean: Easy `6.064`, Medium `5.832`, Hard `7.642`
- Compared with GBS128, GBS256 is lower on all splits and has more empty predictions, so it is not the current best SFT batch-size setting for VP.

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
bash exps/eval/run_visualprobe_qwen35_full_eval.sh
```

The default variant is the SFT/RL-aligned final-answer prompt:

```bash
bash exps/eval/run_visualprobe_qwen35_full_eval.sh final_sentence
```

For the plain-question prompt:

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
MODEL_PATH=/path/to/Qwen3.5-9B-or-SFT-checkpoint bash exps/eval/run_visualprobe_qwen35_full_eval.sh final_sentence
MINIO3_EVAL_FOREGROUND=1 bash exps/eval/run_visualprobe_qwen35_full_eval.sh final_sentence
RAY_NUM_CPUS=128 AGENT_NUM_WORKERS=128 bash exps/eval/run_visualprobe_qwen35_full_eval.sh final_sentence
```

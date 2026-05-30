# Qwen3.5-9B Official-Tool Cold-Start SFT

This records the Qwen3.5-9B SFT setup for Mini-o3 official zoom-tool conversations.

## Data

- Source: `data/minio3_coldstart_hf`
- Converted shards: `data/minio3_coldstart_verl_sft_qwen35_official_tool/train_shards`
- Rows: 7267
- Shards: 15
- Size: 6.5G

Rebuild:

```bash
rm -rf data/minio3_coldstart_verl_sft_qwen35_official_tool/train_shards
uv run --project . --no-sync python examples/minio3/preprocess_coldstart_sft.py \
  --input data/minio3_coldstart_hf \
  --output data/minio3_coldstart_verl_sft_qwen35_official_tool/train_shards \
  --rows-per-shard 512 \
  --min-pixels 40000 \
  --max-pixels 2000000
```

The converter writes byte-image parquet shards without embedding `min_pixels` or `max_pixels` into the image struct. The SFT dataset injects those values through `data.image_min_pixels=40000` and `data.image_max_pixels=2000000`. This avoids pyarrow failures on a single large nested byte-image parquet.

## Launch

Stable launcher:

```bash
bash exps/train/run_qwen35_official_tool_h200_sft.sh formal
```

Frozen launchers:

```text
exps/train/run_qwen35_official_tool_h200_sft_lora_20260529.sh
exps/train/run_qwen35_official_tool_h200_sft_full_freeze_20260530.sh
exps/train/run_qwen35_official_tool_h200_sft_full_freeze_gbs128_20260530.sh
```

Use the frozen launcher for any run that should remain comparable with a
specific experiment version. The moving launcher
`run_qwen35_official_tool_h200_sft.sh` keeps the current default profile.

Current stable defaults:

```text
MODEL_PATH=/mnt/localssd/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
FINETUNING_TYPE=lora
TRAIN_BATCH_SIZE=32
MICRO_BATCH_SIZE_PER_GPU=1
MAX_LENGTH=32768
MAX_TOKEN_LEN_PER_GPU=32768
USE_DYNAMIC_BSZ=True
SP_SIZE=1
FSDP_STRATEGY=fsdp2
LORA_RANK=8
LORA_ALPHA=16
LORA_TARGET_MODULES=.*model\.language_model\.layers\..*\.mlp\.(gate_proj|up_proj|down_proj)$
LR=1e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE=cosine
TOTAL_EPOCHS=3
SAVE_FREQ=100
ADD_VISION_ID=True
WHOLE_CONVERSATION_TOKENIZE=True
READ_PARQUET_DTYPE_BACKEND=default
FREEZE_VISION_TOWER=False
FREEZE_MULTI_MODAL_PROJECTOR=False
```

Official Mini-o3 reference uses Qwen2.5-VL-7B-Instruct, cold-start SFT data, image pixels 40000/2000000, cutoff 32768, lr 1e-5, 3 epochs, cosine schedule, warmup 0.1, bf16, and frozen vision/projector. Local adaptation keeps the core length/pixel/lr/epoch schedule but uses Qwen3.5-9B official tool-call formatting and language-side LoRA to align with the later RL/vLLM path. Reference: https://github.com/Mini-o3/Mini-o3

Full-tuning probe mode:

```bash
FINETUNING_TYPE=full bash exps/train/run_qwen35_official_tool_h200_sft.sh formal
```

In `FINETUNING_TYPE=full`, the wrapper sets `LORA_RANK=0` and follows the
official Mini-o3 SFT freeze policy by default:

```text
FREEZE_VISION_TOWER=True
FREEZE_MULTI_MODAL_PROJECTOR=True
```

For Qwen3.5 HF models this freezes `model.visual` and its `merger` projector
before FSDP wrapping, leaving `model.language_model` and `lm_head` trainable.

## Runs

### qwen35_9b_official_tool_h200_sft_20260529_235552

- Status: completed 681/681 steps.
- Frozen launcher: `exps/train/run_qwen35_official_tool_h200_sft_lora_20260529.sh`
- Tmux: `minio3_sft_formal_20260529_235552` completed and exited.
- Log: `logs/qwen35_9b_official_tool_h200_sft_20260529_235552.log`
- Save dir: `save/qwen35_9b_official_tool_h200_sft_20260529_235552`
- Record dir: `artifacts/train/qwen35_9b_official_tool_h200_sft_20260529_235552`
- W&B: https://wandb.ai/dddraxxx/Mini-o3-qwen35-sft/runs/3irqvyui
- Total steps: 681
- Warmup steps: 68
- First 8 steps: loss 0.6676-0.7222, grad norm 0.2204-0.2807, global tokens about 108k-141k per step.
- Peak memory by step 8: about 64.94GB allocated and 100.11GB reserved.
- Final step: loss 0.47966, grad norm 0.23977, lr 0, global tokens 95604.
- Final checkpoint: `save/qwen35_9b_official_tool_h200_sft_20260529_235552/global_step_681`

### qwen35_9b_official_tool_h200_sft_full_freeze_20260530_101126

- Status: completed 681/681 steps.
- Frozen launcher: `exps/train/run_qwen35_official_tool_h200_sft_full_freeze_20260530.sh`
- Purpose: repeat SFT with full language-model tuning instead of LoRA.
- Freeze policy: `FREEZE_VISION_TOWER=True`, `FREEZE_MULTI_MODAL_PROJECTOR=True`.
- LoRA: disabled with `LORA_RANK=0`, `LORA_ALPHA=0`.
- Intended comparison target: `qwen35_9b_official_tool_h200_sft_20260529_235552`.
- Tmux: `minio3_sft_formal_20260530_101126` completed and exited.
- Log: `logs/qwen35_9b_official_tool_h200_sft_full_freeze_20260530_101126.log`
- Save dir: `save/qwen35_9b_official_tool_h200_sft_full_freeze_20260530_101126`
- Final FSDP checkpoint: `save/qwen35_9b_official_tool_h200_sft_full_freeze_20260530_101126/global_step_681`
- Merged HF checkpoint for vLLM/eval: `save/qwen35_9b_official_tool_h200_sft_full_freeze_20260530_101126/global_step_681_hf`
- W&B: https://wandb.ai/dddraxxx/Mini-o3-qwen35-sft/runs/49umvrl4
- Loss summary: final step loss 0.39707, whole-run mean 0.43213, last-100 mean 0.39614, last-50 mean 0.39571, last-20 mean 0.39916, last-10 mean 0.40117.
- Epoch mean losses: epoch 1 0.48712, epoch 2 0.41361, epoch 3 0.39564.
- Final step metrics: grad norm 1.76562, lr 0, total tokens 0.080253B.
- Peak memory at final step: 72.62GB allocated, 137.16GB reserved.

### qwen35_9b_official_tool_h200_sft_full_freeze_gbs128_tok48k

- Status: next-run candidate; not launched yet.
- Frozen launcher: `exps/train/run_qwen35_official_tool_h200_sft_full_freeze_gbs128_20260530.sh`
- Purpose: test the paper-style larger effective SFT batch on the local 8x H200 node.
- Main change from `qwen35_9b_official_tool_h200_sft_full_freeze_20260530_101126`: `TRAIN_BATCH_SIZE=128`.
- With 7267 rows, 8-way DP, and dataloader drop-last behavior, expected steps are about 56 per epoch and 168 total for 3 epochs.
- `USE_DYNAMIC_BSZ=True` means `MICRO_BATCH_SIZE_PER_GPU` is not the main per-forward memory knob in this code path. The actual dynamic micro-batch sizing is controlled by `MAX_TOKEN_LEN_PER_GPU`.
- Candidate token budget is `MAX_TOKEN_LEN_PER_GPU=49152`. Current full-freeze run at 32768 has reached about 69.55GB peak allocated and 136.53GB reserved on H200; 49152 is a conservative first increase, while 65536 is likely too close to the memory ceiling for a first formal attempt.
- `MICRO_BATCH_SIZE_PER_GPU=2` is recorded for compatibility and for a possible non-dynamic fallback, but under dynamic batching the real effect comes from the token budget above.
- LR/schedule remains `LR=1e-5`, cosine, warmup 0.1, 3 epochs. Do not scale LR just because the batch is larger unless we intentionally run a separate learning-rate ablation.
- `SAVE_FREQ=50`, giving checkpoints around steps 50, 100, 150, and final.

### qwen35_9b_official_tool_h200_sft_20260529_235314

- Status: failed at first training batch.
- Cause: `SP_SIZE=2` produced a 2x mismatch between `logits_rmpad` and `temperature_rmpad` in `prepare_model_outputs`.
- Fix: use `SP_SIZE=1` for this SFT path and keep `MAX_TOKEN_LEN_PER_GPU=32768`.

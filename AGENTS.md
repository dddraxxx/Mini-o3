# Repository Guidelines

## Project Structure & Module Organization
- `verl/` — core Python package: `trainer/` (Hydra + Ray entrypoints), `workers/`, `models/`, `utils/`, `tools/`, `single_controller/`, `protocol.py`, `version/`.
- `tests/` — pytest suites (e.g., `utility/`, `gpu_utility/`, `rollout/`).
- `scripts/` — helper scripts (e.g., `preprocess_coldstart.py`, `model_merger.py`).
- `docs/`, `examples/`, `assets/`, `docker/`, `sft_configs/`, `patches/` — supplementary materials.
- Root configs: `pyproject.toml`, `setup.py`, `.style.yapf`, `requirements.txt`.

## Build, Test, and Development Commands
- Assumption: Install steps in `README.md` are already completed and the `minio3` conda env is active (`conda activate minio3`). If setting up fresh, follow README first.
- Environment (preferred): conda `minio3` — `conda create -n minio3 python=3.10 -y && conda activate minio3`. Alternative: `python -m venv .venv && source .venv/bin/activate`.
- Install package: `pip install -e .` (or `pip install -r requirements.txt`).
- Dev/test extras: `pip install -e .[test]`.
- Run tests: `pytest -q` (subset: `pytest tests/utility -q`).
- Format code: `yapf -ir verl tests scripts`.
- Run training locally: `python -m verl.trainer.main_ppo …` with Hydra CLI overrides (see `README.md` and `verl/trainer/config/`).

## Coding Style & Naming Conventions
- Python ≥3.8, 4-space indentation, 120-char limit (see `.style.yapf`, Google style).
- Naming: modules/functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- Prefer type hints and Google-style docstrings. Use `logging` (not `print`).
- Keep changes minimal and localized; avoid unrelated refactors.

## Testing Guidelines
- Place tests under `tests/` with filenames `test_*.py`.
- Aim for deterministic, fast unit tests; isolate GPU/Ray-heavy tests.
- Include fixtures under `tests/**`; do not commit large assets/checkpoints.
- New public APIs require tests. Run `pytest -q` before pushing.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (≤72 chars), optional scope (e.g., `trainer:`).
- PRs include: clear summary, rationale, before/after behavior, test results, and repro commands.
- Link issues (`Fixes #123`) and update docs/configs/examples when behavior changes.
- Run formatter and tests; keep diffs focused and reviewable.

## Security & Configuration Tips
- Never commit secrets, API keys, or large model files; `.gitignore` covers common artifacts.
- Use environment variables for credentials (e.g., `API_KEY`) and configure Ray/Hydra via CLI or YAML.
- Prefer paths under project directories (e.g., `./save/`) for outputs and logs.

## Local HF Cache & Training Env
- All Hugging Face caches are configured to live under `./data` to avoid writes to the home directory.
- A ready-to-source `.env` is provided at the repo root with the following variables:
  - `HF_HOME=./data/hf_home`
  - `HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`
  - `HF_DATASETS_CACHE=./data/hf_datasets_cache`
  - `TRANSFORMERS_CACHE=$HF_HOME/transformers`
  - `WANDB_DISABLED=true` (disable Weights & Biases by default)

Usage (from repo root):

```bash
conda activate minio3
source .env
```

These settings ensure models and datasets download into `./data` only.

## Local Tips & Gotchas
- LlamaFactory SFT training (with `cdd` helper): `FORCE_TORCHRUN=1 cdd 2 llamafactory-cli train sft_configs/qwen2.5-vl.yaml` (this is the config used for Qwen2.5-VL SFT). The `cdd` helper from bashrc expands to running the command with `CUDA_VISIBLE_DEVICES` set to the first argument, so the above is equivalent to: `CUDA_VISIBLE_DEVICES=2 FORCE_TORCHRUN=1 llamafactory-cli train sft_configs/qwen2.5-vl.yaml`. You can pass lists like `cdd 0,1 ...` to target multiple GPUs.
- Coldstart preprocessing: `--output_dir` must be an absolute path. Example: `python3 scripts/preprocess_coldstart.py --dataset_path Mini-o3/Mini-o3-Coldstart-Dataset --output_dir $(pwd)/data/coldstart_out`

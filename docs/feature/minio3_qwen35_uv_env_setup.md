# Mini-o3 Qwen3.5 uv Env Setup

This note records the repo-local Qwen3.5 official/H200 environment setup added
for Mini-o3. The goal is to make the current working environment reproducible
without letting a normal resolver pass downgrade or break the Qwen3.5 stack.

## Result

The Qwen3.5 profile is kept outside the root `pyproject.toml`:

```bash
requirements/minio3-qwen35-official.in
requirements/minio3-qwen35-official.txt
requirements/minio3-qwen35-official-overrides.txt
requirements/README.md
examples/minio3/install_qwen35_official_env.sh
```

The installer still uses staged installs because the official stack cannot be
represented as a plain dependency solve today: `vllm==0.18.0` still declares
`transformers>=4.56.0,<5`, while the official Qwen3.5 path requires the
Transformers 5.x Qwen3.5 commit:

```bash
cc7ab9be508ce6ed3637bba9e50367b29b742dc6
```

`requirements/minio3-qwen35-official-overrides.txt` is therefore required for
both compile and install commands.

## Current Pinned Stack

Important pinned packages:

| Package | Version |
| --- | --- |
| `torch` | `2.10.0` selected with `--torch-backend cu128` |
| `vllm` | `0.18.0` |
| `transformers` | git commit `cc7ab9be508ce6ed3637bba9e50367b29b742dc6` |
| `flashinfer-python` | `0.6.6` |
| `flashinfer-jit-cache` | `0.6.6+cu129` |
| `flash-attn` | `2.8.3` |
| `causal-conv1d` | `1.6.2.post1` |
| `flash-linear-attention` | `0.5.0` |
| `qwen-vl-utils` | `0.0.14` |
| `pandas` | `3.0.3` |

The checked-in lock intentionally omits local `+cu128` tags from `torch`,
`torchaudio`, and `torchvision` so uv sees the already installed packages as
matching. CUDA 12.8 wheel selection comes from `--torch-backend cu128`.

## Regenerate

Use the profile and override file together:

```bash
uv pip compile requirements/minio3-qwen35-official.in \
  --overrides requirements/minio3-qwen35-official-overrides.txt \
  --python .venv/bin/python \
  --torch-backend cu128 \
  --index https://flashinfer.ai/whl/cu129 \
  --index-strategy unsafe-best-match \
  --no-emit-package verl \
  --output-file requirements/minio3-qwen35-official.txt
```

If the generated lock is meant to preserve the current `.venv` rather than
upgrade transitive packages, constrain from a current freeze first and then
normalize the torch local tags:

```bash
uv pip freeze --python .venv/bin/python | rg -v '^-e ' > /tmp/minio3-qwen35-current-freeze.txt

uv pip compile requirements/minio3-qwen35-official.in \
  --constraints /tmp/minio3-qwen35-current-freeze.txt \
  --overrides requirements/minio3-qwen35-official-overrides.txt \
  --python .venv/bin/python \
  --torch-backend cu128 \
  --index https://flashinfer.ai/whl/cu129 \
  --index-strategy unsafe-best-match \
  --no-emit-package verl \
  --output-file requirements/minio3-qwen35-official.txt
```

## Install

Use the staged installer:

```bash
bash examples/minio3/install_qwen35_official_env.sh
```

The installer applies:

```bash
--constraints requirements/minio3-qwen35-official.txt
--overrides requirements/minio3-qwen35-official-overrides.txt
--index https://flashinfer.ai/whl/cu129
--index-strategy unsafe-best-match
--torch-backend cu128
```

It still installs the official Transformers commit with `--no-deps` after vLLM,
then installs the newer `huggingface-hub` and `typer` requirements needed by
that commit.

## Validation

Lightweight checks used for this setup:

```bash
bash -n examples/minio3/install_qwen35_official_env.sh

uv pip install --dry-run --python .venv/bin/python \
  --constraints requirements/minio3-qwen35-official.txt \
  --overrides requirements/minio3-qwen35-official-overrides.txt \
  --index https://flashinfer.ai/whl/cu129 \
  --index-strategy unsafe-best-match \
  --torch-backend cu128 \
  "vllm==0.18.0" \
  "qwen-vl-utils>=0.0.14" \
  "pandas>=2.3.0,<4"

uv run --python .venv/bin/python --no-sync python \
  examples/minio3/check_qwen35_env.py \
  --model-path Qwen/Qwen3.5-9B \
  --local-files-only
```

Expected dry-run result against the current `.venv`:

```text
Would make no changes
```

Expected env check versions:

```text
transformers=5.3.0.dev0 vllm=0.18.0 torch=2.10.0+cu128
```

## Why Not Root uv.lock

A root project lock is not the right minimum change here. `uv lock --dry-run`
currently tries to solve the project extras together and fails because
`verl[vllm]` requires the vLLM torch line while `verl[sglang]` requires a
different torch version. Keeping this as a dedicated requirements profile avoids
rewriting upstream verl dependency metadata and keeps the Qwen3.5 workaround
localized.

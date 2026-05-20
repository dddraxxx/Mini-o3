# Mini-o3 requirements profiles

`minio3-qwen35-official.in` is the reproducible Qwen3.5/H200 environment
profile. It is separate from the root `pyproject.toml` so upstream verl's
dynamic dependency and extras structure can remain intact.

Regenerate the pinned lock with:

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

Install with:

```bash
bash examples/minio3/install_qwen35_official_env.sh
```

Do not remove the `--overrides` file from compile or install commands until
vLLM publishes metadata that allows the official transformers 5.x Qwen3.5
commit.

After installing the transformers git commit, the installer reapplies the
lock-pinned `accelerate>=1.13.0` package through the same lock/overrides
profile. Keep the transformers and accelerate pins together when updating this
Qwen3.5 environment.

The checked-in lock omits local `+cu128` tags from `torch`, `torchaudio`, and
`torchvision` so it matches the package metadata in the existing `.venv`.
Installation still selects CUDA 12.8 wheels through `--torch-backend cu128`.

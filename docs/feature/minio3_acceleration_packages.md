# Mini-o3 Acceleration Package Install Notes

本文记录 Qwen3.5 / Mini-o3 训练环境里实测可用的加速包安装方式，方便之后重建
`.venv-qwen35-upstream` 或排查 attention backend。

## 当前实测环境

当前工作树：`/mnt/localssd/Mini-o3-qwen35-9b`

当前 venv：

```bash
.venv-qwen35-upstream
```

当前实测版本：

| 包 / import 名 | 版本 | 用途 |
| --- | --- | --- |
| `flash-attn` / `flash_attn` | `2.8.3` | `flash_attention_2` backend |
| `causal-conv1d` / `causal_conv1d` | `1.6.2.post1` | linear attention 依赖 |
| `flash-linear-attention` / `fla` | `0.5.0` | Qwen3.5 linear attention fast path |
| `fla-core` | `0.5.0` | `flash-linear-attention` 依赖 |
| `qwen-vl-utils` / `qwen_vl_utils` | `0.0.14` | Qwen VL image/video preprocessing |
| `torch` | `2.10.0+cu128` | CUDA `12.8` build |

注意：`fla` 是 import 名，不是 pip distribution 名；pip 里要装的是
`flash-linear-attention`，它会带上 `fla-core`。

## 推荐安装命令

在 repo 根目录执行：

```bash
cd /mnt/localssd/Mini-o3-qwen35-9b

uv pip install --python .venv-qwen35-upstream/bin/python \
  ninja packaging wheel setuptools

uv pip install --python .venv-qwen35-upstream/bin/python \
  --no-build-isolation \
  flash-attn causal-conv1d

uv pip install --python .venv-qwen35-upstream/bin/python \
  flash-linear-attention qwen-vl-utils
```

`flash-attn` 和 `causal-conv1d` 使用 `--no-build-isolation`，避免 build env 里拿不到
当前 venv 的 torch / CUDA 组合。`flash-linear-attention` 当前环境下直接安装即可。

## 验证命令

```bash
cd /mnt/localssd/Mini-o3-qwen35-9b

VIRTUAL_ENV=.venv-qwen35-upstream uv run --active --no-sync python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)

import flash_attn
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))

import causal_conv1d
print("causal_conv1d", getattr(causal_conv1d, "__version__", "unknown"))

import fla
print("fla", getattr(fla, "__version__", "unknown"))

import qwen_vl_utils
print("qwen_vl_utils", getattr(qwen_vl_utils, "__version__", "unknown"))
PY
```

期望至少看到：

```text
torch 2.10.0+cu128
cuda 12.8
flash_attn 2.8.3
causal_conv1d 1.6.2.post1
fla 0.5.0
qwen_vl_utils 0.0.14
```

也可以用 distribution 名确认：

```bash
uv pip list --python .venv-qwen35-upstream/bin/python | grep -Ei 'flash|linear|causal|fla|qwen'
```

## 训练脚本相关开关

Qwen3.5 训练默认优先用 flash attention：

```bash
MODEL_ATTN_IMPLEMENTATION=flash_attention_2
```

如果 flash attention 不可用，可临时回退：

```bash
MODEL_ATTN_IMPLEMENTATION=sdpa
```

linear attention fast path 依赖 `causal_conv1d` 和 `flash-linear-attention`。如果二者缺失，
Transformers / vLLM 仍可能 fallback 到 torch 实现，但速度会差很多。

## 常见坑

- 不要执行 `uv pip install fla`；正确包名是 `flash-linear-attention`。
- 编译类包要在目标 venv 已有匹配 torch 后安装。
- 不要把 HF token 写进脚本或文档。需要访问 Hugging Face 时用 shell 环境变量注入。
- 如果 `flash-attn` 编译失败，先确认 `ninja`、`packaging`、`wheel`、`setuptools` 已安装，
  再用同一个 venv 的 Python 执行 `uv pip install --no-build-isolation flash-attn`。

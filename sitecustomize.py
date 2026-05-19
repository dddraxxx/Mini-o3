"""Process-wide Mini-o3 runtime patches.

This module is intentionally inert unless an opt-in environment variable is set.
Python imports ``sitecustomize`` during interpreter startup when it is available
on ``PYTHONPATH``; that makes it a reliable place to patch vLLM spawn workers.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _patch_vllm_dummy_lora() -> None:
    try:
        from vllm.v1.worker import lora_model_runner_mixin
    except Exception:
        return

    mixin_cls = getattr(lora_model_runner_mixin, "LoRAModelRunnerMixin", None)
    if mixin_cls is None:
        return

    current = getattr(mixin_cls, "maybe_dummy_run_with_lora", None)
    if getattr(current, "_minio3_noop_dummy_lora", False):
        return

    @contextmanager
    def _noop_dummy_lora(self, *args, **kwargs):
        yield

    _noop_dummy_lora._minio3_noop_dummy_lora = True
    mixin_cls.maybe_dummy_run_with_lora = _noop_dummy_lora
    sys.stderr.write("[minio3-stage] sitecustomize patched vLLM V1 dummy LoRA activation\n")
    sys.stderr.flush()


if _env_truthy("VERL_VLLM_SKIP_DUMMY_LORA"):
    _patch_vllm_dummy_lora()

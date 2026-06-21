# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import tempfile
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import collect_lora_params, fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank
from verl.utils.transformers_compat import drop_tied_target_keys, get_auto_model_for_vision2seq

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

LORA_ADAPTER_DIR = "lora_adapter"
LORA_ADAPTER_WEIGHTS = "adapter_model.safetensors"


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
        trust_remote_code: Whether to trust_remote_code when loading the model configuration
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self.trust_remote_code = trust_remote_code

    def _get_peft_model(self):
        model = getattr(self.model, "_fsdp_wrapped_module", self.model)
        return model if hasattr(model, "peft_config") else None

    def _get_adapter_name(self) -> str:
        peft_model = self._get_peft_model()
        if peft_model is None:
            return "default"
        if "default" in peft_model.peft_config:
            return "default"
        return next(iter(peft_model.peft_config))

    @staticmethod
    def _lora_adapter_dir(local_path: str) -> str:
        return os.path.join(local_path, LORA_ADAPTER_DIR)

    @classmethod
    def _has_lora_adapter(cls, local_path: str) -> bool:
        return os.path.exists(os.path.join(cls._lora_adapter_dir(local_path), LORA_ADAPTER_WEIGHTS))

    @staticmethod
    def _insert_adapter_name(key: str, adapter_name: str) -> str:
        for marker in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"):
            token = f".{marker}."
            if token not in key:
                continue
            prefix, suffix = key.split(token, 1)
            if suffix.startswith(f"{adapter_name}."):
                return key
            return f"{prefix}.{marker}.{adapter_name}.{suffix}"
        return key

    def _summon_full_params(self, writeback: bool = False):
        if fsdp_version(self.model) > 0:
            return FSDP.summon_full_params(self.model, writeback=writeback)
        return nullcontext()

    def _save_lora_adapter(self, local_path: str):
        peft_model = self._get_peft_model()
        if peft_model is None:
            raise RuntimeError("save_lora_only=True requires an actor model wrapped by PEFT/LoRA.")

        lora_state_dict = collect_lora_params(self.model, layered_summon=True, base_sync_done=True)
        if not lora_state_dict:
            lora_state_dict = collect_lora_params(self.model, layered_summon=False, base_sync_done=True)

        if self.rank == 0:
            if not lora_state_dict:
                raise RuntimeError("No LoRA tensors found while saving LoRA-only checkpoint.")

            from safetensors.torch import save_file

            adapter_name = self._get_adapter_name()
            adapter_path = self._lora_adapter_dir(local_path)
            local_mkdir_safe(adapter_path)

            peft_config = peft_model.peft_config[adapter_name]
            base_config = getattr(peft_model, "config", None)
            base_model_name = getattr(base_config, "name_or_path", None) or getattr(base_config, "_name_or_path", None)
            if base_model_name and not getattr(peft_config, "base_model_name_or_path", None):
                peft_config.base_model_name_or_path = base_model_name
            peft_config.save_pretrained(adapter_path)

            state_dict = {
                name: tensor.detach().cpu().contiguous() for name, tensor in lora_state_dict.items()
            }
            with tempfile.NamedTemporaryFile(dir=adapter_path, delete=False, suffix=".safetensors") as tmp_weights:
                tmp_weights_name = tmp_weights.name
            try:
                save_file(state_dict, tmp_weights_name)
                os.replace(tmp_weights_name, os.path.join(adapter_path, LORA_ADAPTER_WEIGHTS))
            finally:
                if os.path.exists(tmp_weights_name):
                    os.remove(tmp_weights_name)

            log_with_rank(
                f"Saved LoRA adapter to {os.path.abspath(adapter_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

        torch.distributed.barrier()

    def _load_lora_adapter(self, local_path: str):
        peft_model = self._get_peft_model()
        if peft_model is None:
            raise RuntimeError(f"Found LoRA adapter checkpoint under {local_path}, but actor model is not PEFT/LoRA.")

        from safetensors.torch import load_file

        adapter_path = self._lora_adapter_dir(local_path)
        weights_path = os.path.join(adapter_path, LORA_ADAPTER_WEIGHTS)
        adapter_name = self._get_adapter_name()
        adapter_state_dict = {
            self._insert_adapter_name(name, adapter_name): tensor
            for name, tensor in load_file(weights_path, device="cpu").items()
        }

        if fsdp_version(self.model) == 2:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=True,
                strict=False,
            )
            load_result = set_model_state_dict(peft_model, adapter_state_dict, options=options)
        else:
            with self._summon_full_params(writeback=True):
                load_result = peft_model.load_state_dict(adapter_state_dict, strict=False)

        log_with_rank(
            f"Loaded LoRA adapter from {weights_path}; missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)}",
            rank=self.rank,
            logger=logger,
        )

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        load_lora_adapter = self.should_load_model and self._has_lora_adapter(local_path)
        load_sharded_model = self.should_load_model and not load_lora_adapter

        local_model_path = None
        local_optim_path = None
        local_extra_state_path = None

        # every rank download its own checkpoint
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False) if load_sharded_model else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if load_sharded_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                self.model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if load_lora_adapter:
            self._load_lora_adapter(local_path)

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:
            try:
                for local_ckpt_path in (local_model_path, local_optim_path, local_extra_state_path):
                    if local_ckpt_path is not None and is_non_local(local_ckpt_path):
                        os.remove(local_ckpt_path)
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        global_step: int = 0,
        max_ckpt_to_keep=None,
        save_lora_only: bool = False,
    ):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        if self.rank == 0:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        save_lora_adapter = self.should_save_model and save_lora_only and self._get_peft_model() is not None
        if self.should_save_model and save_lora_only and not save_lora_adapter:
            log_with_rank(
                "save_lora_only=True was requested, but the actor is not PEFT/LoRA; saving full model shards.",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

        save_sharded_model = self.should_save_model and not save_lora_adapter

        # every rank will save its own model and optim shard
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False) if save_sharded_model else None
        )
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if save_sharded_model:
                    model_state_dict = self.model.state_dict()
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if save_lora_adapter:
            self._save_lora_adapter(local_path)

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    auto_model_cls = get_auto_model_for_vision2seq()
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(
                        model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.trust_remote_code
                    )

                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                drop_tied_target_keys(state_dict, save_model, model_config)

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)

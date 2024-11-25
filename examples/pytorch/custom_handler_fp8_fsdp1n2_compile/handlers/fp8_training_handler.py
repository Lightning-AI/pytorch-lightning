# the script is modified based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/float8.py
import logging
import operator
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torch.nn as nn
from lightning_utilities.core.imports import compare_version

log = logging.getLogger(__name__)


def is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


# check https://github.com/pytorch/ao/blob/main/torchao/float8/config.py for more config details
@dataclass
class FP8Config:
    enable_fp8: bool = True
    enable_amax_init: bool = False
    scaling_type_input: str = "delayed"
    scaling_type_weight: str = "delayed"
    scaling_type_grad_output: str = "delayed"
    enable_fsdp_float8_all_gather: bool = False
    precompute_float8_dynamic_scale_for_fsdp: bool = False
    pad_inner_dim: bool = True
    emulate_fp8: bool = False  # Set to True for testing without FP8 hardware
    enable_torch_compile: bool = True
    enable_pre_and_post_forward: bool = False


# Define a map for module filter functions based on model name
MODULE_FILTER_MAP = {
    "llama": lambda mod, fqn: isinstance(mod, nn.Linear) and "mlp" in fqn and "lm_head" not in fqn,
    "mixtral": lambda mod, fqn: isinstance(mod, nn.Linear)
    and "block_sparse_moe" in fqn
    and "block_sparse_moe.gate" not in fqn
    and "lm_head" not in fqn,
    "default": lambda mod, fqn: isinstance(mod, nn.Linear),  # Default filter
}


class Float8TrainingHandler:
    """Handler for configuring models for FP8 training using torchao."""

    def __init__(self, args: FP8Config, model_path: str, parallel_dims: Dict[str, bool]):
        """Initializes the handler for FP8 training and configuration.

        Args:
            args (FP8Config): Configuration object for FP8 training, including settings for scaling, amax initialization, and torch compile.
            model_path (str): The path to the model. Typically used for determining model-specific settings.
            parallel_dims (Dict[str, bool]): Dictionary specifying parallelization settings, such as whether DP shard is enabled.

        Example Usage:
            fp8_config = FP8Config(
                enable_fp8=True,
                enable_amax_init=True,
                scaling_type_input="delayed",
                scaling_type_weight="delayed",
                scaling_type_grad_output="delayed",
                enable_fsdp_float8_all_gather=False,
                precompute_float8_dynamic_scale_for_fsdp=False,
                pad_inner_dim=True,
                emulate_fp8=False,  # Set to True for testing without FP8 hardware
                enable_torch_compile=True,
                enable_pre_and_post_forward=False,
            )

            parallel_dims = {"dp_shard_enabled": False}
            handler = Float8TrainingHandler(fp8_config, "path/to/model", parallel_dims)

        """
        self.model_path = model_path
        self.args = args
        self.parallel_dims = parallel_dims
        self.compile = args.enable_torch_compile
        self.enable_fp8 = args.enable_fp8

        if not self.enable_fp8:
            log.warning("Fp8 is disabled here")
            return

        if not is_sm89_or_later() and not args.emulate_fp8:
            log.error("Failed to swap to Float8Linear because float8 is only supported on SM89 or later (H100+ GPUs)")
            raise RuntimeError("Float8Linear operation is not supported on the current hardware.")

        # Check if torchao is installed and version is >= 0.5.0
        try:
            compare_version("torchao", operator.ge, "0.6.1")
            from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType
        except ImportError as e:
            log.error(str(e))
            raise

        # Configure Float8LinearConfig parameters from args
        scaling_type_input = ScalingType(args.scaling_type_input)
        scaling_type_weight = ScalingType(args.scaling_type_weight)
        scaling_type_grad_output = ScalingType(args.scaling_type_grad_output)

        enable_fsdp_float8_all_gather = (
            parallel_dims.get("dp_shard_enabled", False) and args.enable_fsdp_float8_all_gather
        )

        enable_amax_init = args.enable_amax_init
        self.config = Float8LinearConfig(
            enable_amax_init=enable_amax_init,
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            cast_config_input=CastConfig(scaling_type=scaling_type_input),
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
            enable_pre_and_post_forward=args.enable_pre_and_post_forward,
            pad_inner_dim=args.pad_inner_dim,
            emulate=args.emulate_fp8,
        )

        # For precompute_float8_dynamic_scale_for_fsdp
        self.precompute_scale = enable_fsdp_float8_all_gather and args.precompute_float8_dynamic_scale_for_fsdp

        # For sync_float8_amax_and_scale_history
        self.delayed_scaling = (
            scaling_type_input == ScalingType.DELAYED
            or scaling_type_weight == ScalingType.DELAYED
            or scaling_type_grad_output == ScalingType.DELAYED
        )
        self._sync_float8_amax_and_scale_history = None

        log.info("Float8 training active")

    def convert_to_float8_training(self, model: nn.Module, module_filter_fn: callable = None):
        """Converts the linear layers of `model` to `Float8Linear` based on a module filter function. Mutates the model
        in place.

        Args:
            model (nn.Module): The model whose layers should be converted.
            module_filter_fn (callable, optional): A function to filter which modules should be replaced.
                Defaults to a model-specific filter based on `model_path`.

        """
        if not self.enable_fp8:
            log.warning("FP8 is disabled, so layers will not be replaced.")
            return

        log.warning("Enabling FP8 Training")

        # Use the provided filter function or select from the map
        if module_filter_fn is None:
            model_path_lower = self.model_path.lower()
            module_filter_fn = next(
                (fn for key, fn in MODULE_FILTER_MAP.items() if key in model_path_lower),
                MODULE_FILTER_MAP["default"],  # Default filter if no match is found
            )

        from torchao.float8 import convert_to_float8_training

        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=module_filter_fn,
        )
        log.info(
            f"Swapped to Float8Linear layers with enable_fsdp_float8_all_gather={self.config.enable_fsdp_float8_all_gather}"
        )

    def precompute_float8_dynamic_scale_for_fsdp(self, model: Union[nn.Module, List[nn.Module]]):
        if not self.enable_fp8 or not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)

    def sync_float8_amax_and_scale_history(self, model: Union[nn.Module, List[nn.Module]]):
        if not self.enable_fp8 or not self.delayed_scaling:
            return

        from torchao.float8 import sync_float8_amax_and_scale_history

        # Cache the compiled function if necessary
        if self._sync_float8_amax_and_scale_history is None:
            if self.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(sync_float8_amax_and_scale_history)
            else:
                self._sync_float8_amax_and_scale_history = sync_float8_amax_and_scale_history

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            self._sync_float8_amax_and_scale_history(m)

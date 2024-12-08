import logging
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning_utilities.core.imports import compare_version

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

log = logging.getLogger(__name__)


@dataclass
class FSDP2Config:
    enable_cpu_offload: bool = False
    enable_gradient_checkpointing: bool = False


class FSDP2Handler:
    """Handler for wrapping the model layers with FSDP2.

    Args:
        args (FSDP2Config): Configuration for FSDP2, including options for CPU offload and gradient checkpointing.
        device_mesh (DeviceMesh): Device mesh configuration for FSDP2 parallelism.

    Attributes:
        args (FSDP2Config): Stores the FSDP2 configuration.
        device_mesh (DeviceMesh): Stores the device mesh configuration.

    """

    def __init__(self, args: FSDP2Config, device_mesh: "DeviceMesh"):
        self.args = args
        self.device_mesh = device_mesh

        # Check PyTorch version for FSDP2 support (currently we require PyTorch >= 2.6.0)
        try:
            compare_version("torch", operator.ge, "2.6.0")
        except RuntimeError as e:
            log.error(str(e))
            raise

        # Import necessary FSDP modules
        try:
            from torch.distributed._composable.fsdp import (
                CPUOffloadPolicy,
                MixedPrecisionPolicy,
                fully_shard,
            )
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
            )

            self.fully_shard = fully_shard
            self.checkpoint_wrapper = checkpoint_wrapper
            self.MixedPrecisionPolicy = MixedPrecisionPolicy
            self.CPUOffloadPolicy = CPUOffloadPolicy
        except ImportError as e:
            log.error(f"Failed to import FSDP modules: {e}")
            raise

    def wrap_model(self, model: nn.Module):
        """Wraps the model layers with FSDP configurations.

        Args:
            model (nn.Module): The model to wrap.

        Returns:
            nn.Module: The wrapped model.

        """
        dp_mesh = self.device_mesh["data_parallel"]
        assert dp_mesh.size() > 1, "FSDP requires at least two devices."

        fsdp_policy = {
            "mesh": dp_mesh,
            "mp_policy": self.MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        }
        if self.args.enable_cpu_offload:
            fsdp_policy["offload_policy"] = self.CPUOffloadPolicy()

        for layer_id, module in enumerate(model.model.layers):
            reshard_after_forward = layer_id < len(model.model.layers) - 1
            if self.args.enable_gradient_checkpointing:
                module = self.checkpoint_wrapper(module)
            self.fully_shard(
                module,
                **fsdp_policy,
                reshard_after_forward=reshard_after_forward,
            )
            model.model.layers[layer_id] = module

        self.fully_shard(model, **fsdp_policy)
        return model

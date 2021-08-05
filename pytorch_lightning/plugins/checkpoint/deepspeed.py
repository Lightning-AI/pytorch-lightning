from pathlib import Path
from typing import Any, Dict, Mapping, Union, Optional

import torch
from torch import Tensor

from pytorch_lightning.plugins.checkpoint.checkpoint import CheckpointPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _DEEPSPEED_AVAILABLE
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()
if _DEEPSPEED_AVAILABLE:
    import deepspeed


class DeepSpeedCheckpointPlugin(CheckpointPlugin):

    def save_checkpoint(self, checkpoint: Dict, filepath: str) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
        """
        if self.training_type_plugin.zero_stage_3 and self.training_type_plugin._multi_device and self.training_type_plugin.is_global_zero:
            warning_cache.warn(
                "When saving the DeepSpeed Stage 3 checkpoint, "
                "each worker will save a shard of the checkpoint within a directory. "
                "If a single file is required after training, "
                "see https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html#"
                "deepspeed-zero-stage-3-single-file for instructions."
            )
        # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
        # dump states as a checkpoint dictionary object
        _exclude_keys = ["state_dict", "optimizer_states", "lr_schedulers"]
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.model.save_checkpoint(filepath, client_state=checkpoint)

    def load_checkpoint_file(self, checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        if self.training_type_plugin.load_full_weights and self.training_type_plugin.zero_stage_3:
            # Broadcast to ensure we load from the rank 0 checkpoint
            # This doesn't have to be the case when using deepspeed sharded checkpointing
            checkpoint_path = self.training_type_plugin.broadcast(checkpoint_path)
            return super().load_checkpoint_file(checkpoint_path)

        # Rely on deepspeed to load the checkpoint and necessary information
        from pytorch_lightning.trainer.states import TrainerFn

        is_fitting = self.lightning_module.trainer.state.fn == TrainerFn.FITTING
        _, client_state = self.model.load_checkpoint(
            checkpoint_path, load_optimizer_states=is_fitting, load_lr_scheduler_states=is_fitting
        )
        if client_state is None:
            raise MisconfigurationException(
                "DeepSpeed was unable to load the checkpoint. Ensure you passed in a DeepSpeed compatible checkpoint "
                "or a single checkpoint file with `Trainer(plugins=DeepSpeedPlugin(load_full_weights=True))`."
            )
        return client_state

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # override to do nothing, deepspeed engine already loaded the weights in `load_checkpoint_file()`
        if self.training_type_plugin.load_full_weights and self.training_type_plugin.zero_stage_3:
            self.training_type_plugin.model_to_device()
            self._restore_zero_state(checkpoint)

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # override to do nothing, deepspeed engine already loaded the states in `load_checkpoint_file()`
        pass

    def lightning_module_state_dict(self) -> Dict[str, Union[Any, Tensor]]:
        """Returns model state."""
        model = self.lightning_module
        return model.state_dict()

    def _restore_zero_state(self, ckpt: Mapping[str, Any]) -> None:
        """
        Overrides the normal load_state_dict behaviour in PyTorch to ensure
        we gather parameters that may be sharded across processes before loading
        the state dictionary when using ZeRO stage 3.
        This is then automatically synced across processes.

        Args:
            ckpt: The ckpt file.
        """

        def load(module: torch.nn.Module, prefix=""):

            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            state_dict = ckpt["state_dict"]

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if self.training_type_plugin.is_global_zero:
                    module._load_from_state_dict(
                        state_dict=state_dict,
                        prefix=prefix,
                        local_metadata=local_metadata,
                        strict=True,
                        missing_keys=missing_keys,
                        unexpected_keys=unexpected_keys,
                        error_msgs=error_msgs,
                    )

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self.lightning_module, prefix="")

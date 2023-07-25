# Copyright The Lightning AI team.
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
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import XLAStrategy
from lightning.fabric.strategies.fsdp import _apply_filter
from lightning.fabric.strategies.strategy import _BackwardSyncControl, _validate_keys_for_strict_loading
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH, Optimizable

from torch_xla.distributed.parallel_loader import MpDeviceLoader


class XLAFSDPStrategy(XLAStrategy):
    """Strategy for training multiple TPU devices using the
    :func:`torch_xla.distributed.xla_fully_sharded_data_parallel.XlaFullyShardedDataParallel` method.

    For more information check out https://github.com/pytorch/xla/blob/master/docs/fsdp.md
    """

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        auto_wrap_policy: Optional[Callable] = None,
        auto_wrapper_callable: Optional[Callable] = None,
        state_dict_type: Literal["full", "sharded"] = "sharded",
        sync_module_states: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        self._backward_sync_control = _XLAFSDPBackwardSyncControl()

        if auto_wrap_policy:
            kwargs["auto_wrap_policy"] = auto_wrap_policy
        if auto_wrapper_callable:
            kwargs["auto_wrapper_callable"] = auto_wrapper_callable

        self._fsdp_kwargs = kwargs
        self._state_dict_type = state_dict_type
        self._sync_module_states = sync_module_states

    def setup_environment(self) -> None:
        from torch_xla.experimental.pjrt import using_pjrt

        assert self.parallel_devices is not None
        if using_pjrt() and len(self.parallel_devices) == 1:
            # spawning only 1 device with PjRT is not supported:
            # https://github.com/Lightning-AI/lightning/pull/17408#discussion_r1170671732
            raise NotImplementedError(
                "The `XLAFSDPStrategy` does not support running on a single device with the PjRT runtime."
                " Try using all devices or the `SingleTPUStrategy` strategy"
            )

        self._launched = True
        rank_zero_only.rank = self.global_rank
        super().setup_environment()

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple[Module, List[Optimizer]]:
        """Returns NotImplementedError since for XLA FSDP optimizer setup must happen after module setup."""
        raise NotImplementedError(
            f"The `{type(self).__name__}` does not support the joint setup of module and optimizer(s)."
            " Please do it in this order: Create the model, call `setup_module`, create the optimizer,"
            " call `setup_optimizer`."
        )

    def setup_module(self, module: Module) -> Module:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        if any(isinstance(mod, XLAFSDP) for mod in module.modules()):
            if "auto_wrap_policy" in self._fsdp_kwargs:
                rank_zero_warn(
                    "A XLAFSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored."
                )
                del self._fsdp_kwargs["auto_wrap_policy"]
        else:
            if self._sync_module_states:
                import torch_xla.core.xla_model as xm

                xm.broadcast_master_param(module)

            module = XLAFSDP(
                module=module,
                **self._fsdp_kwargs,
            )

        return module

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Set up an optimizer for a model wrapped with XLAFSDP.

        This setup method doesn't modify the optimizer or wrap the optimizer. The only thing it currently does is verify
        that the optimizer was created after the model was wrapped with :meth:`setup_module` with a reference to the
        flattened parameters.
        """
        if _TORCH_GREATER_EQUAL_2_0:
            return optimizer

        from torch_xla.distributed.fsdp.xla_flatten_params_wrapper import FlatParameter

        num_groups = len(optimizer.param_groups)
        if num_groups > 1:
            raise ValueError(
                "An optimizer used with an XLA FSDP model does not support multiple param groups."
                f" Found {num_groups} parameter groups."
            )

        if any(isinstance(param, FlatParameter) for param in optimizer.param_groups[0]["params"]):
            return optimizer

        raise ValueError(
            "The optimizer does not seem to reference any XLA FSDP parameters. HINT: Make sure to create the optimizer"
            " after setting up the model."
        )

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        """Overrides default tpu optimizer_step since FSDP should not call
        `torch_xla.core.xla_model.optimizer_step`. Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            **kwargs: Any extra arguments to ``optimizer.step``
        """
        return optimizer.step(**kwargs)

    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    def process_dataloader(self, dataloader: DataLoader) -> "MpDeviceLoader":
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        if isinstance(dataloader, MpDeviceLoader):
            # dataloader is already wrapped by MpDeviceLoader
            return dataloader

        dataloader = MpDeviceLoader(dataloader, self.root_device)
        # Mimic interface to torch.utils.data.DataLoader
        dataloader.dataset = dataloader._loader.dataset
        dataloader.batch_sampler = getattr(dataloader._loader, "batch_sampler", None)
        return dataloader

    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        """Clip gradients by norm."""
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        if not isinstance(module, XLAFSDP):
            # the root must be wrapped
            raise TypeError(
                "Gradient clipping with XLAFSDP is only possible if the module passed to"
                f" `{self.__class__.__name__}.clip_gradients_norm` is wrapped in `XLAFullyShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        self.precision.unscale_gradients(optimizer)
        return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)

    def clip_gradients_value(  # type: ignore[override]
        self, module: Module, optimizer: Optimizer, clip_val: Union[float, int]
    ) -> None:
        """Clip gradients by value."""
        raise NotImplementedError(
            "XLA's FSDP strategy does not support to clip gradients by value."
            " Consider clipping by norm instead or choose another strategy!"
        )

    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state in a checkpoint directory.

        The directory will contain one file per process, with model- and optimizer shards stored per file. Additionally,
        it creates a a consolidated checkpoint combining all of the sharded checkpoints.
        """
        if "shard_metadata" not in state:
            raise TypeError(
                "Saving and loading checkpoints with `XLAFSDPStrategy` requires the state to include shard_metadata."
                'Please add `"shard_metadata": model._forward_module.get_shard_metadata()` to `state`.'
            )

        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `XLAFSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`."
            )
        if storage_options is not None:
            raise TypeError(
                "`XLAFSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `XLAFSDPStrategy` does not use the `CheckpointIO`."
            )
        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = self.broadcast(path)
        if Path(path).is_dir() and os.listdir(path):
            raise FileExistsError(f"The checkpoint directory already exists and is not empty: {path}")
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        modules = [module for module in state.values() if isinstance(module, XLAFSDP)]
        if len(modules) == 0:
            raise ValueError(
                "Could not find a XLA FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple XLA FSDP modules in the given state. Saving checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )
        import torch_xla.core.xla_model as xm

        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        converted_state: Dict[str, Any] = {}
        for key, obj in state.items():
            # convert the state
            if isinstance(obj, Module) and isinstance(obj, XLAFSDP):
                converted = obj.state_dict()
            elif isinstance(obj, Optimizer):
                converted = obj.state_dict()
            else:
                converted = obj
            _apply_filter(key, filter or {}, converted, converted_state)

        self.checkpoint_io.save_checkpoint(
            converted_state, f"{path}_rank-{rank:08d}-of-{world_size:08d}.pth", storage_options=storage_options
        )

        if self._state_dict_type == "full":
            from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints

            if xm.is_master_ordinal(local=False):
                consolidate_sharded_model_checkpoints(
                    ckpt_prefix=str(path),
                    ckpt_suffix="_rank-*-of-*.pth",
                )
            xm.rendezvous("ckpt_consolidation")

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint
        """
        import torch_xla.core.xla_model as xm

        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        filepath = Path(self.broadcast(filepath))
        if filepath.is_file():
            raise NotImplementedError(
                "The `XLAFSDPStrategy` requires specifying the directory where to remove checkpoints."
            )

        self.checkpoint_io.remove_checkpoint(f"{filepath}_rank-{rank}-of-{world_size}.pth")
        if self._state_dict_type == "full":
            self.checkpoint_io.remove_checkpoint(f"{filepath}_consolidated.pth")

    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load the contents from a checkpoint and restore the state of the given objects.

        The strategy currently only supports saving and loading sharded checkpoints which are stored in form of a
        directory of multiple files rather than a single file.
        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `FSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`."
            )
        if not state:
            raise ValueError(
                f"Got XLAFSDPStrategy.load_checkpoint(..., state={state!r}) but a state with at least "
                f" a model instance to reload is required. Pass it in like so:"
                " FSDPStrategy.load_checkpoint(..., state={'model': model, ...})"
            )

        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = self.broadcast(path)
        if Path(path).is_file():
            raise NotImplementedError(
                f"The path `{path}` is a file, but the `XLAFSDPStrategy` currently only supports loading from a"
                " checkpoint with sharded states in a directory."
            )
        import torch_xla.core.xla_model as xm

        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        file = f"{path}_rank-{rank:08d}-of-{world_size:08d}.pth"
        if not Path(file).is_file():
            raise ValueError(
                f"The path {str(file)!r} does not point to a valid checkpoint. Make sure the path points to a"
                " directory with XLA FSDP checkpoint shards."
            )
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        modules = {key: module for key, module in state.items() if isinstance(module, XLAFSDP)}
        optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
        if self._state_dict_type == "sharded":
            if len(modules) == 0:
                raise ValueError(
                    "Could not find a XLA FSDP model in the provided checkpoint state. Please provide the model as"
                    " part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure"
                    " you set up the model (and optimizers if any) through the strategy before loading the checkpoint."
                )
            if len(modules) > 1:
                raise ValueError(
                    "Found multiple XLA FSDP modules in the given state. Loading checkpoints with FSDP is"
                    " currently limited to a single model per checkpoint. To load multiple models, call the"
                    " load method for each model separately with a different path."
                )

            _, module = list(modules.items())[0]
            sharded_ckpt = torch.load(file)

            module.load_state_dict(sharded_ckpt["model"], strict=strict)
            for opt_key, opt in optimizers.items():
                opt.load_state_dict(sharded_ckpt[opt_key])

            # Load anything leftover from sharded_ckpt
            loaded_metadata_keys = sharded_ckpt.keys() - modules.keys() - optimizers.keys()
            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, loaded_metadata_keys, strict=strict)
            for key in requested_metadata_keys:
                if key in loaded_metadata_keys:
                    state[key] = sharded_ckpt[key]
                    loaded_metadata_keys.remove(key)

            metadata = {}
            if len(loaded_metadata_keys):
                for key in loaded_metadata_keys:
                    metadata[key] = sharded_ckpt[key]

            return metadata
        elif self._state_dict_type == "full":
            rank_zero_warn(
                "Loading a full checkpoint will only load the full model."
                "Optimizer and any additional metadata are not included."
            )
            if len(modules) > 0:
                raise ValueError(
                    "Found a XLA FSDP model in the provided checkpoint state."
                    "Please provide the model without any FSDP wrapper."
                )
            full_ckpt = torch.load(f"{str(path)}_consolidated.pth")
            state["model"].load_state_dict(full_ckpt["model"], strict=strict)
        else:
            raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register("xla_fsdp", cls, description=cls.__class__.__name__)


class _XLAFSDPBackwardSyncControl(_BackwardSyncControl):
    @contextmanager
    def no_backward_sync(self, module: Module) -> Generator:
        """Blocks gradient synchronization inside the
        :class:`~torch_xla.distributed.fsdp.XlaFullyShardedDataParallel` wrapper."""
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        if not isinstance(module, XLAFSDP):
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{self.__class__.__name__}.no_backward_sync` is wrapped in `XlaFullyShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        with module.no_sync():
            yield

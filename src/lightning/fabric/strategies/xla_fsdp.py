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
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING, Union

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
from lightning.fabric.strategies.strategy import _BackwardSyncControl
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH, Optimizable

if TYPE_CHECKING and _XLA_AVAILABLE:
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
else:
    XLAFSDP = None
    MpDeviceLoader = None


class XLAFSDPStrategy(XLAStrategy):
    """Strategy for training multiple TPU devices using the
    :func:`torch_xla.distributed.xla_fully_sharded_data_parallel.XlaFullyShardedDataParallel` method."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        self._backward_sync_control = _XLAFSDPBackwardSyncControl()
        self._fsdp_kwargs = kwargs

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

        if "auto_wrap_policy" in self._fsdp_kwargs and any(isinstance(mod, XLAFSDP) for mod in module.modules()):
            # If model is already wrapped, we need to avoid sending the `auto_wrap_policy`
            del self._fsdp_kwargs["auto_wrap_policy"]

        from torch_xla.experimental import pjrt

        pjrt.broadcast_master_param(module)

        wrapped_module = XLAFSDP(
            module=module,
            **self._fsdp_kwargs,
        )

        return wrapped_module

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

    def clip_gradients_norm(  # type: ignore[override]
        self,
        module: "XLAFSDP",
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        groups: Optional[List[List[int]]] = None,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        """Clip gradients by norm."""
        rank_zero_warn("Gradient Clipping by Norm is currently experimental for XLA FSDP. Proceed with Caution!")
        self.precision.unscale_gradients(optimizer)
        return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type, groups=groups)

    def clip_gradients_value(  # type: ignore[override]
        self, module: "XLAFSDP", optimizer: Optimizer, clip_val: Union[float, int]
    ) -> None:
        """Clip gradients by value."""
        raise NotImplementedError(
            "XLA's FSDP strategy does not support to clip gradients by value."
            " Consider clipping by norm instead or choose another strategy!"
        )

    def save_checkpoint(
        self, path: _PATH, state: Dict[str, Union[Module, Optimizer, Any]], storage_options: Optional[Any] = None
    ) -> None:
        """Save model, optimizer, and other state in a checkpoint directory.

        The directory will contain one file per process, with model- and optimizer shards stored per file. Additionally,
        it creates a a consolidated checkpoint combining all of the sharded checkpoints.
        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(
                "Saving and loading checkpoints with the `FSDPStrategy` is not supported in PyTorch < 2.0."
                " Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`."
            )
        if storage_options is not None:
            raise TypeError(
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `FSDPStrategy` does not use the `CheckpointIO`."
            )
        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(path))
        if path.is_dir() and os.listdir(path):
            raise FileExistsError(f"The checkpoint directory already exists and is not empty: {path}")

        modules = [module for module in state.values() if isinstance(module, XLAFSDP)]
        if len(modules) == 0:
            raise ValueError(
                "Could not find a FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple FSDP modules in the given state. Saving checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )

        import torch_xla.core.xla_model as xm

        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        state = self._convert_stateful_objects_in_state(state)

        self.checkpoint_io.save_checkpoint(
            state, f"{path}_rank-{rank}-of-{world_size}.pth", storage_options=storage_options
        )

        from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints

        if xm.is_master_ordinal(local=False):
            consolidate_sharded_model_checkpoints(
                ckpt_prefix="/tmp/mnist-fsdp-fabric/final_ckpt", ckpt_suffix="_rank-*-of-*.pth"
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
        self.checkpoint_io.remove_checkpoint(f"{filepath}_consolidated.pth")

    def load_checkpoint(
        self, path: _PATH, state: Optional[Dict[str, Union[Module, Optimizer, Any]]] = None
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
                f"Got FSDPStrategy.load_checkpoint(..., state={state!r}) but a state with at least "
                f" a model instance to reload is required. Pass it in like so:"
                " FSDPStrategy.load_checkpoint(..., state={'model': model, ...})"
            )

        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(path))
        if path.is_file():
            raise NotImplementedError(
                f"The path `{path}` is a file, but the `XLAFSDPStrategy` currently only supports loading from a"
                " checkpoint with sharded states in a directory."
            )

        modules = {key: module for key, module in state.items() if isinstance(module, XLAFSDP)}
        optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
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

        import torch_xla.core.xla_model as xm

        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        sharded_ckpt = torch.load(f"{path}_rank-{rank}-of-{world_size}.pth")

        module.load_state_dict(sharded_ckpt["model"])
        for opt_key, opt in optimizers.items():
            opt.load_state_dict(sharded_ckpt[opt_key])

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


def _optimizer_has_flat_params(optimizer: Optimizer) -> bool:
    from torch_xla.distributed.fsdp.xla_flatten_params_wrapper import FlatParameter

    return any(isinstance(param, FlatParameter) for param in optimizer.param_groups[0]["params"])

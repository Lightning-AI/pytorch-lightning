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
import io
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.accelerators.tpu import _XLA_AVAILABLE
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.io.xla import XLACheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import ParallelStrategy
from lightning.fabric.strategies.launchers.xla import _XLALauncher
from lightning.fabric.strategies.strategy import TBroadcast
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_2_0,
)
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH, ReduceOp

if TYPE_CHECKING and _XLA_AVAILABLE:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader


class FSDPXLAStrategy(ParallelStrategy):
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
            cluster_environment=XLAEnvironment(),
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        self._checkpoint_io: Optional[CheckpointIO]
        self._backward_sync_control = None  # XLA synchronizes gradients in the optimizer.step() call
        self._launched = False
        self._fsdp_kwargs = kwargs

        # if _TORCH_GREATER_EQUAL_2_0:
        # Enables joint setup of model and optimizer, multiple optimizer param groups, and `torch.compile()`
        # self._fsdp_kwargs.setdefault("use_orig_params", True)

    @property
    def root_device(self) -> torch.device:
        if not self._launched:
            raise RuntimeError("Accessing the XLA device before processes have spawned is not allowed.")
        import torch_xla.core.xla_model as xm

        return xm.xla_device()

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = XLACheckpointIO()
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    def _configure_launcher(self) -> None:
        self._launcher = _XLALauncher(self)

    def setup_environment(self) -> None:
        from torch_xla.experimental.pjrt import using_pjrt

        assert self.parallel_devices is not None
        if using_pjrt() and len(self.parallel_devices) == 1:
            # spawning only 1 device with PjRT is not supported:
            # https://github.com/Lightning-AI/lightning/pull/17408#discussion_r1170671732
            raise NotImplementedError(
                "The `FSDPXLAStrategy` does not support running on a single device with the PjRT runtime."
                " Try using all devices or the `SingleTPUStrategy` strategy"
            )

        self._launched = True
        rank_zero_only.rank = self.global_rank
        super().setup_environment()

    def setup_module(self, module: Module) -> Module:
        if "auto_wrap_policy" in self._fsdp_kwargs and any(isinstance(mod, FSDP) for mod in module.modules()):
            # If model is already wrapped, we need to avoid sending the `auto_wrap_policy`
            del self._fsdp_kwargs["auto_wrap_policy"]

        wrapped_module = FSDP(
            module=module,
            **self._fsdp_kwargs,
        )

        from torch_xla.experimental import pjrt

        pjrt.broadcast_master_param(wrapped_module)
        return wrapped_module

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Set up an optimizer for a model wrapped with FSDP.

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
                "An optimizer used with an FSDP XLA model does not support multiple param groups."
                f" Found {num_groups} parameter groups."
            )

        if any(isinstance(param, FlatParameter) for param in optimizer.param_groups[0]["params"]):
            return optimizer

        raise ValueError(
            "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the optimizer"
            " after setting up the model."
        )

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

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Function to gather a tensor from several distributed processes.

        Args:
            tensor: tensor to all-gather.
            group: unused.
            sync_grads: flag that allows users to synchronize gradients for the all-gather operation.
        Return:
            A tensor of shape (world_size, ...)
        """
        if not self._launched:
            return tensor
        if not isinstance(tensor, Tensor):
            raise NotImplementedError(
                f"`{type(self).__name__}.all_gather` is only implemented for tensors. Given {tensor}"
            )
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.device.type != "xla":
            tensor = tensor.to(self.root_device)

        import torch_xla.core.functions as xf
        import torch_xla.core.xla_model as xm

        return xf.all_gather(tensor) if sync_grads else xm.all_gather(tensor)

    def all_reduce(
        self, output: Union[Tensor, Any], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> Tensor:
        if not isinstance(output, Tensor):
            output = torch.tensor(output, device=self.root_device)

        invalid_reduce_op = isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if invalid_reduce_op or invalid_reduce_op_str:
            raise ValueError(
                "Currently, the FSDPXLAStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )
        import torch_xla.core.xla_model as xm

        output = xm.mesh_reduce("reduce", output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        if not self._launched:
            return
        import torch_xla.core.xla_model as xm

        if name is None:
            # `None` is not supported: "TypeError: _xla_rendezvous(): incompatible function arguments"
            name = ""
        xm.rendezvous(name)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not self._launched:
            return obj

        import torch_xla.core.xla_model as xm

        is_tensor = isinstance(obj, Tensor)
        if is_tensor:
            if obj.dim() == 0:
                obj = obj.unsqueeze(0)
            if obj.device.type != "xla":
                obj = obj.to(self.root_device)
        else:
            # support for arbitrary pickle-ables
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            obj = torch.tensor(  # type: ignore[assignment]
                bytearray(buffer.getbuffer()), device=self.root_device, dtype=torch.float
            )

        obj = [obj]
        xm.collective_broadcast(obj, root_ordinal=src)
        obj = obj[0]

        if not is_tensor:
            buffer = io.BytesIO(obj.cpu().byte().numpy())
            obj = torch.load(buffer)

        return obj

    def save_checkpoint(
        self, path: _PATH, state: Dict[str, Union[Module, Optimizer, Any]], storage_options: Optional[Any] = None
    ) -> None:
        """Save model, optimizer, and other state as a checkpoint file.

        Args:
            path: A path to where the file(s) should be saved
            state: A dictionary with contents to be saved. If the dict contains modules or optimizers, their
                state-dict will be retrieved and converted automatically.
            storage_options: Additional options for the ``CheckpointIO`` plugin
        """
        state = self._convert_stateful_objects_in_state(state)
        # `xla_model.save` needs to be called on all ranks. It internally checks if the local rank is 0
        self.checkpoint_io.save_checkpoint(state, path, storage_options=storage_options)

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint
        """
        if self.local_rank == 0:
            self.checkpoint_io.remove_checkpoint(filepath)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register("fsdp_xla", cls, description=cls.__class__.__name__)

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
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from lightning_fabric.accelerators import Accelerator
from lightning_fabric.accelerators.tpu import _XLA_AVAILABLE
from lightning_fabric.plugins.environments import XLAEnvironment
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.io.xla import XLACheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies import ParallelStrategy
from lightning_fabric.strategies.launchers.xla import _XLALauncher
from lightning_fabric.strategies.strategy import TBroadcast
from lightning_fabric.utilities.apply_func import apply_to_collection
from lightning_fabric.utilities.data import has_len
from lightning_fabric.utilities.rank_zero import rank_zero_only
from lightning_fabric.utilities.types import _PATH, ReduceOp

if TYPE_CHECKING and _XLA_AVAILABLE:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader


class XLAStrategy(ParallelStrategy):
    """Strategy for training multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn`
    method."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
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

    @property
    def _is_distributed(self) -> bool:
        import torch_xla.core.xla_env_vars as xenv

        # HOST_WORLD_SIZE is not set outside the xmp.spawn process
        return (xenv.HOST_WORLD_SIZE in os.environ) and self.world_size != 1

    def _configure_launcher(self) -> None:
        self._launcher = _XLALauncher(self)

    def setup_environment(self) -> None:
        self._launched = True
        self._set_world_ranks()
        rank_zero_only.rank = self.global_rank
        super().setup_environment()

    def setup_module(self, module: Module) -> Module:
        return module

    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    def process_dataloader(self, dataloader: DataLoader) -> "MpDeviceLoader":
        XLAStrategy._validate_dataloader(dataloader)
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
            tensor: tensor of shape (batch, ...)
            group: not available with TPUs
            sync_grads: flag that allows users to synchronize gradients for the all_gather operation
        Return:
            A tensor of shape (world_size, batch, ...)
        """
        if isinstance(tensor, Tensor) and tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

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
                "Currently, the XLAStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )
        import torch_xla.core.xla_model as xm

        output = xm.mesh_reduce("reduce", output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        if self._is_distributed:
            import torch_xla.core.xla_model as xm

            xm.rendezvous(name)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not self._is_distributed:
            return obj
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        data_tensor = torch.tensor(data, device=self.root_device, dtype=torch.float)
        import torch_xla.core.xla_model as xm

        data = xm.all_gather(data_tensor)
        buffer = io.BytesIO(data.cpu().byte().numpy())
        obj = torch.load(buffer)
        return obj

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
            storage_options: parameter for how to save to storage, passed to ``CheckpointIO`` plugin
        """
        # `xla_model.save` needs to be called on all ranks. It internally checks if the local rank is 0
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint
        """
        if self.local_rank == 0:
            self.checkpoint_io.remove_checkpoint(filepath)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        # TODO(fabric): Deprecate the name "tpu_spawn" through the connector
        strategy_registry.register("tpu_spawn", cls, description=cls.__class__.__name__)
        strategy_registry.register("xla", cls, description=cls.__class__.__name__)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        rank_zero_only.rank = self.cluster_environment.global_rank()

    @staticmethod
    def _validate_dataloader(dataloaders: DataLoader) -> None:
        def check_has_len(dataloader: DataLoader) -> None:
            if not has_len(dataloader):
                raise TypeError(
                    "TPUs do not currently support IterableDataset objects, the dataset must implement `__len__`."
                    " HINT: You can mock the length on your dataset to bypass this MisconfigurationException."
                )

        apply_to_collection(dataloaders, dtype=object, wrong_dtype=(Sequence, Mapping), function=check_has_len)

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
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch.nn import Module

import lightning.pytorch as pl
from lightning.fabric.accelerators.tpu import _XLA_AVAILABLE
from lightning.fabric.plugins import CheckpointIO, XLACheckpointIO
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.data import has_len
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.types import _PATH, ReduceOp
from lightning.pytorch.overrides.base import _LightningModuleWrapperBase
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.launchers.xla import _XLALauncher
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import find_shared_parameters, set_shared_parameters
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT

if TYPE_CHECKING and _XLA_AVAILABLE:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
else:
    MpDeviceLoader = None


class XLAStrategy(DDPStrategy):
    """Strategy for training multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn`
    method."""

    strategy_name = "xla"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
        **_: Any,
    ) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=XLAEnvironment(),
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            start_method="fork",
        )
        self._checkpoint_io: Optional[CheckpointIO]
        self.debug = debug
        self._launched = False

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = XLACheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = XLACheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def root_device(self) -> torch.device:
        if not self._launched:
            raise RuntimeError("Accessing the XLA device before processes have spawned is not allowed.")
        import torch_xla.core.xla_model as xm

        return xm.xla_device()

    @property
    def local_rank(self) -> int:
        return self.cluster_environment.local_rank() if self.cluster_environment is not None else 0

    @staticmethod
    def _validate_dataloader(dataloader: object) -> None:
        if not has_len(dataloader):
            raise TypeError(
                "TPUs do not currently support IterableDataset objects, the dataset must implement `__len__`."
                " HINT: You can mock the length on your dataset to bypass this error."
            )

    def connect(self, model: "pl.LightningModule") -> None:
        import torch_xla.distributed.xla_multiprocessing as xmp

        self.wrapped_model = xmp.MpModelWrapper(_LightningModuleWrapperBase(model))
        return super().connect(model)

    def _configure_launcher(self) -> None:
        self._launcher = _XLALauncher(self)

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator
        self.accelerator.setup(trainer)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = "1"

        assert self.lightning_module
        shared_params = find_shared_parameters(self.lightning_module)
        self.model_to_device()

        set_shared_parameters(self.lightning_module, shared_params)
        self.setup_precision_plugin()

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
            _optimizers_to_device(self.optimizers, self.root_device)

    def _setup_model(self, model: Module) -> Module:  # type: ignore
        return model

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    @property
    def is_distributed(self) -> bool:
        # HOST_WORLD_SIZE is not set outside the xmp.spawn process
        import torch_xla.core.xla_env_vars as xenv

        return (xenv.HOST_WORLD_SIZE in os.environ) and self.world_size != 1

    def process_dataloader(self, dataloader: object) -> "MpDeviceLoader":
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

    def configure_ddp(self) -> None:
        pass

    def model_to_device(self) -> None:
        self.model = self.wrapped_model.to(self.root_device)

    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        if self.is_distributed:
            import torch_xla.core.xla_model as xm

            xm.rendezvous(name)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not self.is_distributed:
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

    def reduce(
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

    def setup_distributed(self) -> None:
        self._launched = True
        self.set_world_ranks()
        rank_zero_only.rank = self.global_rank

    def set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        rank_zero_only.rank = self.cluster_environment.global_rank()

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            return self.model(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            return self.model(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            return self.model(*args, **kwargs)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self._pod_progress_bar_force_stdout()

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

    def teardown(self) -> None:
        super().teardown()
        os.environ.pop("PT_XLA_DEBUG", None)

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("xla_debug", cls, description="XLA strategy with `debug` as True", debug=True)
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def _pod_progress_bar_force_stdout(self) -> None:
        # Why is it required? The way `pytorch_xla.distributed` streams logs
        # from different vms to the main worker doesn't work well with tqdm
        # Ref: https://github.com/pytorch/xla/blob/master/torch_xla/distributed/xla_dist.py#L140
        # The print statement seems to force tqdm to flush stdout.
        import torch_xla.core.xla_env_vars as xenv

        if self.global_rank == 0 and int(os.getenv(xenv.TPUVM_MODE, 0)) == 1:
            print()

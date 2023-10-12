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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module

import lightning.pytorch as pl
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE, _XLA_GREATER_EQUAL_2_1, _using_pjrt
from lightning.fabric.plugins import XLACheckpointIO
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.types import _PATH, ReduceOp
from lightning.pytorch.plugins import XLAPrecisionPlugin
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.launchers.xla import _XLALauncher
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import find_shared_parameters, set_shared_parameters
from lightning.pytorch.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader


class XLAStrategy(DDPStrategy):
    """Strategy for training multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn`
    method."""

    strategy_name = "xla"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]] = None,
        precision_plugin: Optional[XLAPrecisionPlugin] = None,
        debug: bool = False,
        sync_module_states: bool = True,
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
        self.debug = debug
        self._launched = False
        self._sync_module_states = sync_module_states

    @property  # type: ignore[override]
    def checkpoint_io(self) -> Union[XLACheckpointIO, _WrappingCheckpointIO]:
        plugin = self._checkpoint_io
        if plugin is not None:
            assert isinstance(plugin, (XLACheckpointIO, _WrappingCheckpointIO))
            return plugin
        return XLACheckpointIO()

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[Union[XLACheckpointIO, _WrappingCheckpointIO]]) -> None:
        if io is not None and not isinstance(io, (XLACheckpointIO, _WrappingCheckpointIO)):
            raise TypeError(f"The XLA strategy can only work with the `XLACheckpointIO` plugin, found {io}")
        self._checkpoint_io = io

    @property  # type: ignore[override]
    def precision_plugin(self) -> XLAPrecisionPlugin:
        plugin = self._precision_plugin
        if plugin is not None:
            assert isinstance(plugin, XLAPrecisionPlugin)
            return plugin
        return XLAPrecisionPlugin()

    @precision_plugin.setter
    def precision_plugin(self, precision_plugin: Optional[XLAPrecisionPlugin]) -> None:
        if precision_plugin is not None and not isinstance(precision_plugin, XLAPrecisionPlugin):
            raise TypeError(
                f"The XLA strategy can only work with the `XLAPrecisionPlugin` plugin, found {precision_plugin}"
            )
        self._precision_plugin = precision_plugin

    @property
    def root_device(self) -> torch.device:
        if not self._launched:
            raise RuntimeError("Accessing the XLA device before processes have spawned is not allowed.")
        import torch_xla.core.xla_model as xm

        return xm.xla_device()

    @property
    def global_rank(self) -> int:
        return super().global_rank if self._launched else 0

    @property
    def local_rank(self) -> int:
        return super().local_rank if self._launched else 0

    @property
    def node_rank(self) -> int:
        return super().node_rank if self._launched else 0

    @property
    def world_size(self) -> int:
        return super().world_size if self._launched else 1

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

        if self._sync_module_states:
            if _XLA_GREATER_EQUAL_2_1:
                from torch_xla.core.xla_model import broadcast_master_param
            else:
                from torch_xla.experimental.pjrt import broadcast_master_param

            broadcast_master_param(self.model)

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
            _optimizers_to_device(self.optimizers, self.root_device)

    def _setup_model(self, model: Module) -> Module:  # type: ignore
        return model

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    def process_dataloader(self, dataloader: object) -> "MpDeviceLoader":
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
        assert self.model is not None
        self.model = self.model.to(self.root_device)

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
            original_device = obj.device
            # XLA distributed requires that the data is on the XLA device
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
            # this will preserve the dtype and device of any tensors
            buffer = io.BytesIO(obj.cpu().byte().numpy())
            obj = torch.load(buffer)
        else:
            obj = obj.to(original_device)

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
        assert self.parallel_devices is not None
        if _using_pjrt() and len(self.parallel_devices) == 1:
            # spawning only 1 device with PjRT is not supported:
            # https://github.com/Lightning-AI/lightning/pull/17408#discussion_r1170671732
            raise NotImplementedError(
                "The `XLAStrategy` does not support running on a single device with the PjRT runtime."
                " Try using all devices or the `SingleDeviceXLAStrategy` strategy"
            )

        self._launched = True
        rank_zero_only.rank = self.global_rank

    def set_world_ranks(self) -> None:
        # accessing global_rank will initialize the XLA computation client. since this is called outside of the spawned
        # processes (by the accelerator connector), we cannot run the code that would normally be here.
        # instead it's done in `setup_distributed`
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        _pod_progress_bar_force_stdout(self.global_rank)

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        import torch_xla.core.xla_model as xm

        # sync any pending lazy tensors on all ranks before saving to prevent potential collective hangs
        xm.mark_step()
        # save on global rank zero only
        super().save_checkpoint(checkpoint, filepath, storage_options=storage_options)

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
        original_device = tensor.device
        tensor = tensor.to(self.root_device)

        import torch_xla.core.functions as xf
        import torch_xla.core.xla_model as xm

        tensor = xf.all_gather(tensor) if sync_grads else xm.all_gather(tensor)
        tensor = tensor.to(original_device)
        return tensor

    def teardown(self) -> None:
        super().teardown()
        self._launched = False  # after the Trainer finishes, we aren't inside the spawned region
        os.environ.pop("PT_XLA_DEBUG", None)

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("xla_debug", cls, description="XLA strategy with `debug` as True", debug=True)
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=cls.__name__,
        )


def _pod_progress_bar_force_stdout(global_rank: int) -> None:
    if _using_pjrt():
        # this was removed in https://github.com/pytorch/xla/pull/5240
        return

    # Why is it required? The way `pytorch_xla.distributed` streams logs
    # from different vms to the main worker doesn't work well with tqdm
    # Ref: https://github.com/pytorch/xla/blob/v2.0.0/torch_xla/distributed/xla_dist.py#L227
    # The print statement seems to force tqdm to flush stdout.
    import torch_xla.core.xla_env_vars as xenv
    from torch_xla.utils.utils import getenv_as

    if global_rank == 0 and getenv_as(xenv.TPUVM_MODE, int, 0) == 1:
        print()

# Copyright The PyTorch Lightning team.
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
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.io.xla_plugin import XLACheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from pytorch_lightning.strategies.launchers.xla_spawn import _XLASpawnLauncher
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _TPU_AVAILABLE, find_shared_parameters, set_shared_parameters
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.optimizer import optimizers_to_device
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import _PATH, STEP_OUTPUT

if _TPU_AVAILABLE:
    import torch_xla.core.xla_env_vars as xenv
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.core.xla_model import rendezvous
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
else:
    xm, xmp, MpDeviceLoader, rendezvous = [None] * 4


class TPUSpawnStrategy(DDPSpawnStrategy):
    """Strategy for training multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn`
    method."""

    strategy_name = "tpu_spawn"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[int]] = None,
        checkpoint_io: Optional[XLACheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
        **_: Any,
    ) -> None:
        checkpoint_io = checkpoint_io or XLACheckpointIO()
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self.debug = debug
        self.tpu_local_core_rank = 0
        self.tpu_global_core_rank = 0
        self.start_method = "fork"

    @property
    def global_rank(self) -> int:
        return self.tpu_global_core_rank

    @property
    def local_rank(self) -> int:
        return self.tpu_local_core_rank

    @property
    def world_size(self) -> int:
        return xm.xrt_world_size()

    @property
    def root_device(self) -> torch.device:
        return xm.xla_device()

    @staticmethod
    def _validate_dataloader(dataloaders: Union[List[DataLoader], DataLoader]) -> None:
        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        for dataloader in dataloaders:
            if not has_len(dataloader):
                raise MisconfigurationException(
                    "TPUs do not currently support IterableDataset objects, the dataset must implement `__len__`."
                    " HINT: You can mock the length on your dataset to bypass this MisconfigurationException."
                )

    @staticmethod
    def _validate_patched_dataloaders(model: "pl.LightningModule") -> None:
        """Validate and fail fast if the dataloaders were passed directly to fit."""
        connector: DataConnector = model.trainer._data_connector
        sources = (
            connector._train_dataloader_source,
            connector._val_dataloader_source,
            connector._test_dataloader_source,
            connector._predict_dataloader_source,
        )
        for source in sources:
            if not source.is_module():
                TPUSpawnStrategy._validate_dataloader(source.instance)

    def connect(self, model: "pl.LightningModule") -> None:
        TPUSpawnStrategy._validate_patched_dataloaders(model)
        self.wrapped_model = xmp.MpModelWrapper(LightningDistributedModule(model))
        return super().connect(model)

    def _configure_launcher(self):
        self._launcher = _XLASpawnLauncher(self)

    def setup(self, trainer: "pl.Trainer") -> None:
        self.start_method = "fork"
        self.accelerator.setup(trainer)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = str(1)

        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        if is_overridden("on_post_move_to_device", self.lightning_module):
            self.model.module.on_post_move_to_device()
        else:
            set_shared_parameters(self.model.module, shared_params)

        self.setup_precision_plugin()

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
            optimizers_to_device(self.optimizers, self.root_device)

    def _setup_model(self, model: Module) -> Module:
        return model

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

    @property
    def is_distributed(self) -> bool:
        # HOST_WORLD_SIZE is None outside the xmp.spawn process
        return os.getenv(xenv.HOST_WORLD_SIZE, None) and self.world_size != 1

    def process_dataloader(self, dataloader: DataLoader) -> MpDeviceLoader:
        TPUSpawnStrategy._validate_dataloader(dataloader)
        dataloader = MpDeviceLoader(dataloader, self.root_device)
        # Mimic interface to torch.utils.data.DataLoader
        dataloader.dataset = dataloader._loader.dataset
        return dataloader

    def configure_ddp(self) -> None:
        pass

    def init_dist_connection(self, global_rank: int, world_size: int) -> None:
        pass

    def set_world_ranks(self, process_idx: int = 0) -> None:
        pass

    def model_to_device(self) -> None:
        self.model = self.wrapped_model.to(self.root_device)

    def barrier(self, name: Optional[str] = None) -> None:
        if self.is_distributed:
            rendezvous(name)

    def broadcast(self, obj: object, src: int = 0) -> object:
        if not self.is_distributed:
            return obj
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        data_tensor = torch.tensor(data, device=self.root_device, dtype=torch.float)
        data = xm.all_gather(data_tensor)
        buffer = io.BytesIO(data.cpu().byte().numpy())
        obj = torch.load(buffer)
        return obj

    def reduce_boolean_decision(self, decision: bool) -> bool:
        decision = torch.tensor(int(decision), device=self.root_device)
        decision = self.reduce(decision, reduce_op="sum")
        decision = bool(decision == self.world_size)
        return decision

    def reduce(self, output, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None):
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output, device=self.root_device)

        _invalid_reduce_op = isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        _invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if _invalid_reduce_op or _invalid_reduce_op_str:
            raise MisconfigurationException(
                "Currently, TPUSpawn Strategy only support `sum`, `mean`, `avg` reduce operation."
            )

        output = xm.mesh_reduce("reduce", output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def _worker_setup(self, process_idx: int):
        reset_seed()
        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()
        rank_zero_only.rank = self.global_rank

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self.model(*args, **kwargs)

    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        self._pod_progress_bar_force_stdout()
        return output

    def validation_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        self._pod_progress_bar_force_stdout()
        return output

    def test_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        self._pod_progress_bar_force_stdout()
        return output

    def _pod_progress_bar_force_stdout(self) -> None:
        # Why is it required? The way `pytorch_xla.distributed` streams logs
        # from different vms to the main worker doesn't work well with tqdm
        # Ref: https://github.com/pytorch/xla/blob/master/torch_xla/distributed/xla_dist.py#L140
        # The print statement seems to force tqdm to flush stdout.
        if self.tpu_global_core_rank == 0 and int(os.getenv(xenv.TPUVM_MODE, 0)) == 1:
            print()

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

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """
        Function to gather a tensor from several distributed processes
        Args:
            tensor: tensor of shape (batch, ...)
            group: not available with TPUs
            sync_grads: not available with TPUs
        Return:
            A tensor of shape (world_size, batch, ...)
        """
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return xm.all_gather(tensor)

    def teardown(self) -> None:
        super().teardown()
        os.environ.pop("PT_XLA_DEBUG", None)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "tpu_spawn_debug", cls, description="TPUSpawn Strategy with `debug` as True", debug=True
        )

        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

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
import re
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.multiprocessing as mp
from torch.nn import Module
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE, _TPU_AVAILABLE, rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.distributed import rank_zero_only, ReduceOp
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import seed_everything

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.core.xla_model import rendezvous
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
else:
    xm, xmp, MpDeviceLoader, rendezvous = [None] * 4

if _OMEGACONF_AVAILABLE:
    from omegaconf import DictConfig, ListConfig, OmegaConf


class TPUSpawnPlugin(DDPSpawnPlugin):
    """ Plugin for training multiple TPU devices using the :func:`torch.multiprocessing.spawn` method. """

    def __init__(self, parallel_devices: Optional[List[int]] = None, **_: Any) -> None:
        super().__init__(parallel_devices, num_nodes=1, cluster_environment=None, sync_batchnorm=False)
        self.tpu_local_core_rank = 0
        self.tpu_global_core_rank = 0
        self.start_method = None

    @property
    def global_rank(self) -> int:
        return self.tpu_local_core_rank

    @property
    def local_rank(self) -> int:
        return self.tpu_local_core_rank

    @property
    def world_size(self) -> int:
        return self.num_processes

    @property
    def root_device(self) -> torch.device:
        return self.device

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
    def _validate_patched_dataloaders(model: Module) -> None:
        """Validate and fail fast if the dataloaders were passed directly to fit.
        """
        if hasattr(model, 'train_dataloader') and isinstance(model.train_dataloader, _PatchDataLoader):
            TPUSpawnPlugin._validate_dataloader(model.train_dataloader.dataloader)

        if hasattr(model, 'val_dataloader') and isinstance(model.val_dataloader, _PatchDataLoader):
            TPUSpawnPlugin._validate_dataloader(model.val_dataloader.dataloader)

        if hasattr(model, 'test_dataloader') and isinstance(model.test_dataloader, _PatchDataLoader):
            TPUSpawnPlugin._validate_dataloader(model.test_dataloader.dataloader)

        if hasattr(model, 'predict_dataloader') and isinstance(model.predict_dataloader, _PatchDataLoader):
            TPUSpawnPlugin._validate_dataloader(model.predict_dataloader.dataloader)

    def connect(self, model: 'pl.LightningModule') -> None:
        TPUSpawnPlugin._validate_patched_dataloaders(model)
        self.wrapped_model = xmp.MpModelWrapper(LightningDistributedModule(model))
        return super().connect(model)

    def setup(self, model: Module) -> Module:
        self.create_mp_queue()
        return self.model

    def create_mp_queue(self):
        self.start_method = 'fork'
        smp = mp.get_context(self.start_method)
        self.mp_queue = smp.SimpleQueue()

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

    @property
    def is_distributed(self) -> bool:
        return self.world_size != 1

    def process_dataloader(self, dataloader: DataLoader) -> MpDeviceLoader:
        TPUSpawnPlugin._validate_dataloader(dataloader)
        return MpDeviceLoader(dataloader, self.device)

    def configure_ddp(self) -> None:
        pass

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        pass

    def set_world_ranks(self, process_idx: int = 0) -> None:
        pass

    def new_process(self, process_idx: int, trainer, mp_queue) -> None:
        self.mp_queue = mp_queue

        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        if self.tpu_global_core_rank != 0 and trainer.progress_bar_callback is not None:
            trainer.progress_bar_callback.disable()

        self.model_to_device()
        trainer.accelerator.setup_optimizers(trainer)
        trainer.precision_plugin.connect(self._model, None, None)

        self.barrier("pre-run-stage")

        results = trainer.run_stage()

        self.transfer_distrib_spawn_state_on_fit_end(results)

        # https://github.com/pytorch/xla/issues/1801#issuecomment-602799542
        self.barrier("end-process")

        # https://github.com/pytorch/xla/issues/2190#issuecomment-641665358
        if self.global_rank == 0:
            time.sleep(2)

    def model_to_device(self) -> None:
        self.device = xm.xla_device()
        self.model = self.wrapped_model.to(self.device)

    def barrier(self, name: Optional[str] = None) -> None:
        rendezvous(name)

    def transfer_distrib_spawn_state_on_fit_end(self, results):
        checkpoint_callback = self.lightning_module.trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

        if self.mp_queue is not None:
            rank_zero_warn("cleaning up ddp environment...")

            # save the last weights
            last_path = None
            if (
                self.lightning_module.trainer.state == TrainerState.FITTING and best_model_path is not None
                and len(best_model_path) > 0
            ):
                last_path = re.sub(".ckpt", ".tmp_end.ckpt", best_model_path)
                self.save(self.lightning_module.state_dict(), last_path)

            if self.global_rank == 0:
                # todo, pass complete checkpoint as state dictionary
                self.mp_queue.put(best_model_path)
                self.mp_queue.put(last_path)
                self.mp_queue.put(results)

    def save(self, state_dict: Dict, path: str) -> None:
        xm.save(state_dict, path)

    def broadcast(self, obj: object, src: int = 0) -> object:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        data_tensor = torch.tensor(data, device=self.device, dtype=torch.float)
        data = xm.all_gather(data_tensor)
        buffer = io.BytesIO(data.cpu().byte().numpy())
        obj = torch.load(buffer)
        return obj

    def reduce_boolean_decision(self, decision: bool) -> bool:
        decision = torch.tensor(int(decision), device=self.lightning_module.device)
        decision = self.reduce(decision, reduce_op="sum")
        decision = bool(decision == self.world_size)
        return decision

    def reduce(self, output, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None):
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output, device=self.lightning_module.device)

        _invalid_reduce_op = isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        _invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if _invalid_reduce_op or _invalid_reduce_op_str:
            raise MisconfigurationException(
                "Currently, TPUSpawn TrainingTypePlugin only support `sum`, `mean`, `avg` reduce operation."
            )

        output = xm.mesh_reduce('reduce', output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def _close_logger(self, trainer) -> None:
        if trainer.logger is not None:
            trainer.logger.finalize("success")

    @property
    def xmp_spawn_kwargs(self):
        return {
            "args": (self.lightning_module.trainer, self.mp_queue),
            "nprocs": len(self.parallel_devices),
            "start_method": self.start_method
        }

    def start_training(self, trainer) -> None:
        # todo: precision pluging is call in accelerator setup and should be moved
        if 'XLA_USE_BF16' in os.environ:
            del os.environ["XLA_USE_BF16"]
        self._close_logger(trainer)
        xmp.spawn(self.new_process, **self.xmp_spawn_kwargs)

    def start_evaluating(self, trainer) -> None:
        self._close_logger(trainer)
        xmp.spawn(self.new_process, **self.xmp_spawn_kwargs)

    def start_predicting(self, trainer) -> None:
        xmp.spawn(self.new_process, **self.xmp_spawn_kwargs)

    def training_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: str) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """
        # Todo: TypeError: 'mappingproxy' object does not support item assignment
        if _OMEGACONF_AVAILABLE:
            checkpoint = apply_to_collection(checkpoint, (DictConfig, ListConfig), OmegaConf.to_container)
        self.save({k: v for k, v in checkpoint.items() if k != "callbacks"}, filepath)

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

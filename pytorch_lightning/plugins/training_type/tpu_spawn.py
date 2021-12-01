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
from multiprocessing.queues import SimpleQueue
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.multiprocessing as mp
from torch.nn import Module
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.io.xla_plugin import XLACheckpointIO
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _TPU_AVAILABLE, find_shared_parameters, rank_zero_warn, set_shared_parameters
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.distributed import rank_zero_only, ReduceOp
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
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


class TPUSpawnPlugin(DDPSpawnPlugin):
    """Plugin for training multiple TPU devices using the :func:`torch.multiprocessing.spawn` method."""

    def __init__(
        self,
        parallel_devices: Optional[List[int]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        debug: bool = False,
        **_: Any
    ) -> None:
        checkpoint_io = checkpoint_io or XLACheckpointIO()
        super().__init__(parallel_devices=parallel_devices, checkpoint_io=checkpoint_io)
        self.debug = debug
        self.tpu_local_core_rank = 0
        self.tpu_global_core_rank = 0
        self.start_method = None

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
                TPUSpawnPlugin._validate_dataloader(source.instance)

    def connect(self, model: "pl.LightningModule") -> None:
        TPUSpawnPlugin._validate_patched_dataloaders(model)
        self.wrapped_model = xmp.MpModelWrapper(LightningDistributedModule(model))
        return super().connect(model)

    def pre_dispatch(self):
        if self.debug:
            os.environ["PT_XLA_DEBUG"] = str(1)

    def setup(self) -> None:
        self.create_mp_queue()

    def _setup_model(self, model: Module) -> Module:
        return model

    def create_mp_queue(self):
        self.start_method = "fork"
        smp = mp.get_context(self.start_method)
        self.mp_queue = smp.SimpleQueue()

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

    @property
    def is_distributed(self) -> bool:
        # HOST_WORLD_SIZE is None outside the xmp.spawn process
        return os.getenv(xenv.HOST_WORLD_SIZE, None) and self.world_size != 1

    def process_dataloader(self, dataloader: DataLoader) -> MpDeviceLoader:
        TPUSpawnPlugin._validate_dataloader(dataloader)
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

    def new_process(self, trainer: "pl.Trainer", mp_queue: SimpleQueue) -> None:
        self.mp_queue = mp_queue

        if self.tpu_global_core_rank != 0 and trainer.progress_bar_callback is not None:
            trainer.progress_bar_callback.disable()

        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        if is_overridden("on_post_move_to_device", self.lightning_module):
            self.model.module.on_post_move_to_device()
        else:
            set_shared_parameters(self.model.module, shared_params)

        trainer.accelerator.setup_optimizers(trainer)
        trainer.precision_plugin.connect(self._model, None, None)

        self.barrier("pre-run-stage")

        results = trainer.run_stage()

        self.__transfer_distrib_spawn_state_on_fit_end(trainer, results)

        # https://github.com/pytorch/xla/issues/1801#issuecomment-602799542
        self.barrier("end-process")

        # https://github.com/pytorch/xla/issues/2190#issuecomment-641665358
        if self.local_rank == 0:
            time.sleep(2)

        # ensure that spawned processes go through teardown before joining
        trainer._call_teardown_hook()

    def model_to_device(self) -> None:
        self.model = self.wrapped_model.to(self.root_device)

    def barrier(self, name: Optional[str] = None) -> None:
        if self.is_distributed:
            rendezvous(name)

    def __transfer_distrib_spawn_state_on_fit_end(self, trainer: "pl.Trainer", results: Any) -> None:
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

        # requires to compute the state_dict on all processes in case Metrics are present
        state_dict = self.lightning_module.state_dict()

        if self.mp_queue is not None:
            rank_zero_warn("cleaning up tpu spawn environment...")

            # save the last weights
            last_path = None
            if trainer.state.fn == TrainerFn.FITTING and best_model_path is not None and len(best_model_path) > 0:
                last_path = re.sub(".ckpt", ".tmp_end.ckpt", best_model_path)
                self.save(state_dict, last_path)

            if self.local_rank == 0:
                # todo, pass complete checkpoint as state dictionary
                self.mp_queue.put(best_model_path)
                self.mp_queue.put(last_path)
                self.mp_queue.put(results)
                self.lightning_module.add_to_queue(self.mp_queue)  # adds the `callback_metrics` to the queue

    def save(self, state_dict: Dict, path: _PATH) -> None:
        xm.save(state_dict, path)

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

        output = xm.mesh_reduce("reduce", output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def get_mp_spawn_kwargs(self, trainer: Optional["pl.Trainer"] = None) -> Dict[str, Any]:
        return {
            "nprocs": len(self.parallel_devices),
            "start_method": self.start_method,
        }

    def spawn(self, function: Callable, *args: Any, return_result: bool = True, **kwargs: Any) -> Optional[Any]:
        context = mp.get_context(self.start_method or "fork")
        return_queue = context.SimpleQueue() if return_result else None
        xmp.spawn(self._wrapped_function, args=(function, args, kwargs, return_queue), **self.get_mp_spawn_kwargs())
        return return_queue.get() if return_result else None

    def _wrapped_function(
        self, process_idx: int, function: Callable, args: Any, kwargs: Any, return_queue: Optional[SimpleQueue]
    ) -> None:
        self._worker_setup(process_idx)
        result = function(*args, **kwargs)
        if return_queue is not None and self.local_rank == 0:
            return_queue.put(move_data_to_device(result, "cpu"))

        self.barrier("end-process")
        # https://github.com/pytorch/xla/issues/2190#issuecomment-641665358
        if self.local_rank == 0:
            time.sleep(2)

    def _worker_setup(self, process_idx: int):
        reset_seed()
        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()
        rank_zero_only.rank = self.global_rank

    def start_training(self, trainer: "pl.Trainer") -> None:
        # todo: precision pluging is call in accelerator setup and should be moved
        if "XLA_USE_BF16" in os.environ:
            del os.environ["XLA_USE_BF16"]
        self._clean_logger(trainer)
        return super().start_training(trainer)

    def start_evaluating(self, trainer: "pl.Trainer") -> None:
        self._clean_logger(trainer)
        return super().start_evaluating(trainer)

    def start_predicting(self, trainer: "pl.Trainer") -> None:
        self._clean_logger(trainer)
        return super().start_predicting(trainer)

    def training_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
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
        # from different vms to the master worker doesn't work well with tqdm
        # Ref: https://github.com/pytorch/xla/blob/master/torch_xla/distributed/xla_dist.py#L140
        # The print statement seems to force tqdm to flush stdout.
        if self.tpu_global_core_rank == 0 and int(os.getenv(xenv.TPUVM_MODE, 0)) == 1:
            print()

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: _PATH) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """
        return self.checkpoint_io.save_checkpoint(checkpoint, filepath)

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
        # TPU teardown
        os.environ.pop("PT_XLA_DEBUG", None)
        self.barrier("teardown")

    @property
    def should_rank_save_checkpoint(self) -> bool:
        return self.local_rank == 0

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register("tpu_spawn_debug", cls, description="TPUSpawn Plugin with `debug` as True", debug=True)

    @property
    def checkpoint_io(self) -> CheckpointIO:
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, plugin: CheckpointIO) -> None:
        raise MisconfigurationException("TPU Spawn Plugin currently does not support custom checkpoint plugins.")

    @staticmethod
    def _clean_logger(trainer: "pl.Trainer") -> None:
        loggers = trainer.logger._logger_iterable if isinstance(trainer.logger, LoggerCollection) else [trainer.logger]
        for logger in loggers:
            if isinstance(logger, TensorBoardLogger) and logger._experiment is not None:
                # the experiment class of `TensorBoard` holds a multiprocessing queue which can make ours hang.
                # we want to make sure these are closed before we spawn our own threads.
                # assuming nothing else references the experiment object, python should instantly `__del__` it.
                logger._experiment = None

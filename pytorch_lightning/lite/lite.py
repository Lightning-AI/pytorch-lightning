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

from abc import abstractmethod, ABC
from collections import Callable
from contextlib import contextmanager
from typing import Any, Optional, Sequence, Union, List

import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.lite.wrappers import _LiteOptimizer, _LiteModel
from pytorch_lightning.plugins import PLUGIN_INPUT
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.utilities import move_data_to_device


class LightningLite(ABC):
    def __init__(
        self,
        accelerator: Optional[Union[str, Accelerator]] = None,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        ipus: Optional[int] = None,
        num_processes: int = 1,
        devices: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        precision: Union[int, str] = 32,
        amp_backend: str = "native",
        amp_level: Optional[str] = None,
        replace_sampler_ddp: bool = True,
    ):
        gpu_ids, tpu_cores = Trainer._parse_devices(gpus=gpus, auto_select_gpus=False, tpu_cores=tpu_cores)
        backend_connector = AcceleratorConnector(
            num_processes=num_processes,
            devices=devices,
            tpu_cores=tpu_cores,
            ipus=ipus,
            distributed_backend=None,  # TODO: remove
            accelerator=accelerator,
            gpus=gpus,
            gpu_ids=gpu_ids,
            num_nodes=num_nodes,
            sync_batchnorm=False,  # TODO: add support?
            benchmark=False,
            replace_sampler_ddp=replace_sampler_ddp,
            deterministic=False,
            precision=precision,
            amp_type=amp_backend,
            amp_level=amp_level,
            plugins=plugins,
        )
        self.accelerator = backend_connector.select_accelerator()

        # TODO: Do we need to initialize distributed at the very beginning
        #    any reason to delay??
        #    this will also launch processes in ddp/ddp_spawn
        self.accelerator.setup_environment()

    @property
    def training_type_plugin(self):
        return self.accelerator.training_type_plugin

    @property
    def precision_plugin(self):
        return self.accelerator.precision_plugin

    @property
    def device(self):
        # the device on the local rank
        return self.training_type_plugin.root_device

    @abstractmethod
    def run(self, *args, **kwarg):
        pass

    def setup(
        self,
        models: Union[nn.Module, Sequence[nn.Module]],
        optimizers: Union[Optimizer, Sequence[Optimizer]],
    ):
        # wrap all objects passed in and return them in the same order
        models = [models] if len(models) == 1 else models
        optimizers = [optimizers] if len(optimizers) == 1 else optimizers
        models, optimizers = self._setup_models_and_optimizers(models, optimizers)

        models = models[0] if len(models) == 1 else models
        optimizers = optimizers[0] if len(optimizers) == 1 else optimizers
        return models, optimizers

    def _setup_models_and_optimizers(self, models: Sequence[nn.Module], optimizers: Sequence[Optimizer]):
        # Let accelerator/plugin wrap and connect the models and optimizers
        models, optimizers = self.training_type_plugin.setup_models_and_optimizers(models, optimizers)
        models = [_LiteModel(module=model, accelerator=self.accelerator) for model in models]
        optimizers = [_LiteOptimizer(optimizer=optimizer, accelerator=self.accelerator) for optimizer in optimizers]
        return models, optimizers

    def setup_dataloader(self, *dataloaders: DataLoader):
        # user can call this method independently instead of the general purpose setup method
        dataloaders = [self.training_type_plugin.setup_dataloader(dataloader) for dataloader in dataloaders]
        dataloaders = dataloaders[0] if len(dataloaders) == 1 else dataloaders
        return dataloaders

    def backward(self, tensor: Tensor, *args, **kwargs):
        # user will call automator.backward(loss) instead of loss.backward()
        self.accelerator.run_backward(tensor, *args, **kwargs)

    @contextmanager
    def forward_context(self):
        with self.accelerator.forward_context():
            yield

    # @contextmanager
    # def backward_context(self, *args, **kwargs):
    #     yield
    #
    # @contextmanager
    def optimizer_step_context(self, model=None, optimizer=None):
        # necessary for deepspeed + scaling
        temp = optimizer.step
        optimizer.step = model.step
        yield
        optimizer.step = temp

    def to_device(self, obj: Union[nn.Module, Tensor]) -> Union[nn.Module, Tensor]:
        if isinstance(obj, nn.Module):
            return obj.to(self.device)
        return move_data_to_device(obj, device=self.device)

    def sync(self, data: Any) -> Any:
        # all_gather
        pass

    def reduce_data(self, data: Any) -> Any:
        return self.training_type_plugin.reduce(data)

    def reduce_decision(self, decision: bool) -> bool:
        return self.training_type_plugin.reduce_boolean_decision(decision)

    def broadcast_decision(self, decision: bool):
        # return self.training_type_plugin.broadcast_boolean_decision(decision)
        return False

    def save_checkpoint(self, filepath):
        pass

    def execute_on_rank(self, func: Callable, rank: int):
        pass

    def spawn(self, function: Callable, *args: Any):
        # ctx = mp.spawn(function, args, nprocs=..., ...)
        pass

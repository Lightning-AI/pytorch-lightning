from weakref import proxy
from collections import Callable
from contextlib import contextmanager
from typing import Any, Union, Optional, Sequence, Tuple

import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    AcceleratorConnector,
)
from pytorch_lightning.utilities import move_data_to_device


class LiteOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, accelerator: Accelerator):
        super().__init__(params=optimizer.param_groups, defaults={})
        self.optimizer = optimizer
        self._accelerator = accelerator

    def step(self, closure=None, **kwargs: Any):
        print("running automated step")
        output = self._accelerator.run_optimizer_step(
            self.optimizer,
            lambda_closure=closure,
            **kwargs,
        )
        return output


class LiteModel(nn.Module):
    def __init__(self, module: nn.Module, accelerator: Accelerator):
        super().__init__()
        self._module = module
        self._accelerator = accelerator

    @property
    def module(self):
        return self._module

    def forward(self, *args, **kwargs):
        with self._accelerator.forward_context():
            output = self.module.forward(*args, **kwargs)
        return output


class LightningLite:
    def __init__(
        self,
        accelerator=None,
        plugins=None,
        gpus=None,
        tpu_cores=None,
        ipus=None,
        num_processes=1,
        devices=None,
        num_nodes=1,
        precision=32,
        amp_backend: str = "native",
        amp_level: str = "O2",
        replace_sampler_ddp=True,
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
        models = [LiteModel(module=model, accelerator=self.accelerator) for model in models]
        optimizers = [LiteOptimizer(optimizer=optimizer, accelerator=self.accelerator) for optimizer in optimizers]
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

from collections import Callable
from contextlib import contextmanager
from typing import Any, Union

import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    AcceleratorConnector,
)
from pytorch_lightning.utilities import move_data_to_device


class AutomatedOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer):
        super().__init__(params=optimizer.param_groups, defaults={})
        self.optimizer = optimizer

    def step(self, closure=None):
        # TODO: do precision magic here
        return self.optimizer.step(closure)


class Automator:
    def __init__(
        self,
        accelerator=None,
        plugin=None,
        gpus=None,
        tpus=None,
        num_processes=None,
        num_nodes=1,
        precision=32,
        amp_backend: str = "native",
        amp_level: str = "O2",
    ):
        backend_connector = AcceleratorConnector(
            gpus=gpus,
            tpu_cores=tpus,
            num_processes=num_processes,
            distributed_backend=accelerator,
            num_nodes=num_nodes,
            precision=precision,
            amp_type=amp_backend,
            amp_level=amp_level,
            plugins=[plugin],
            # TODO:
            deterministic=False,
            sync_batchnorm=False,
            benchmark=False,
            replace_sampler_ddp=True,
            auto_select_gpus=False,
        )
        self.accelerator = backend_connector.select_accelerator()

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

    def setup(self, *objects: Union[nn.Module, Optimizer, DataLoader]):
        # wrap all objects passed in and return them in the same order
        wrapped_objects = []
        for obj in objects:
            if isinstance(obj, nn.Module):
                wrapped_objects.extend(self.setup_model(obj))
            if isinstance(obj, Optimizer):
                wrapped_objects.extend(self.setup_optimizer(obj))
            if isinstance(obj, DataLoader):
                wrapped_objects.extend(self.setup_dataloader(obj))

        if len(wrapped_objects) == 1:
            return wrapped_objects[0]
        return wrapped_objects

    def setup_model(self, *models: nn.Module):
        # user can call this method independently instead of the general purpose setup method
        return [self.training_type_plugin.setup_model(model) for model in models]

    def setup_optimizer(self, *optimizers: Optimizer):
        # user can call this method independently instead of the general purpose setup method
        # TODO: let plugin setup optimizer too?
        return [AutomatedOptimizer(optimizer) for optimizer in optimizers]

    def setup_dataloader(self, *dataloaders: DataLoader):
        # user can call this method independently instead of the general purpose setup method
        return [
            self.training_type_plugin.setup_dataloader(dataloader)
            for dataloader in dataloaders
        ]

    def backward(self, tensor: Tensor, *args, **kwargs):
        # TODO: precision plugin backward
        return tensor.backward(*args, **kwargs)

    @contextmanager
    def forward_context(self):
        # basically only for autocast and block ddp sync
        yield

    @contextmanager
    def backward_context(self, *args, **kwargs):
        # necessary for deepspeed backward + scaler in AMP
        yield

    @contextmanager
    def optimizer_step_context(self, *args, **kwargs):
        # necessary for deepspeed + scaling
        yield

    def to_device(self, obj: Union[nn.Module, Tensor]) -> Union[nn.Module, Tensor]:
        if isinstance(obj, nn.Module):
            return obj.to(self.device)
        return move_data_to_device(obj, device=self.device)

    def sync(self, data: Any) -> Any:
        pass

    def reduce_data(self, data: Any) -> Any:
        pass

    def reduce_decision(self, decision: bool):
        return False

    def broadcast_decision(self, decision: bool):
        return False

    def save_checkpoint(self, filepath):
        pass

    def execute_on_rank(self, func: Callable, rank: int):
        pass

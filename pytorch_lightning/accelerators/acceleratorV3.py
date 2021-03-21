from collections import Callable
from typing import Any, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.plugins import DDPPlugin, PrecisionPlugin
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.utilities import move_data_to_device


class AcceleratedOptimizer(Optimizer):

    def __init__(self, optimizer: Optimizer):
        super().__init__(params=optimizer.param_groups, defaults={})
        self.optimizer = optimizer

    def step(self, closure=None):
        # TODO: do precision magic here
        return self.optimizer.step(closure)


class AcceleratorV3:

    def __init__(self):
        # hardcoded for a start
        # this also needs to incorporate some of the accelerator connectors logic for argument handling
        self.training_type_plugin = DDPPlugin(
            parallel_devices=[torch.device("cuda", 0), torch.device("cuda", 1)],
            num_nodes=1,
            cluster_environment=LightningEnvironment(),
            sync_batchnorm=False,
        )
        self.precision_plugin = PrecisionPlugin()

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
        return [AcceleratedOptimizer(optimizer) for optimizer in optimizers]

    def setup_dataloader(self, *dataloaders: DataLoader):
        # user can call this method independently instead of the general purpose setup method
        return [self.training_type_plugin.setup_dataloader(dataloader) for dataloader in dataloaders]

    def backward(self, tensor: Tensor, *args, **kwargs):
        # TODO: precision plugin backward
        return tensor.backward(*args, **kwargs)

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
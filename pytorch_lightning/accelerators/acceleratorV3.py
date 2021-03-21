from collections import Callable
from typing import Any, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.plugins import DDPPlugin, PrecisionPlugin
from pytorch_lightning.plugins.environments import LightningEnvironment


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

    def setup(self, *objects: Union[nn.Module, Optimizer, DataLoader]):
        # wrap all objects passed in and return them in the same order
        wrapped_objects = []
        for obj in objects:
            if isinstance(obj, nn.Module):
                wrapped_objects.append(self.setup_model(obj))
            if isinstance(obj, Optimizer):
                wrapped_objects.append(self.setup_optimizer(obj))
            if isinstance(obj, DataLoader):
                wrapped_objects.append(self.setup_dataloader(obj))
        return wrapped_objects

    def setup_model(self, model: nn.Module):
        # user can call this method independently instead of the general purpose setup method
        pass

    def setup_optimizer(self, *optimizers: Optimizer):
        # user can call this method independently instead of the general purpose setup method
        pass

    def setup_dataloader(self, *dataloaders: DataLoader):
        # user can call this method independently instead of the general purpose setup method
        pass

    def sync(self, data: Any) -> Any:
        pass

    def reduce_data(self, data: Any) -> Any:
        pass

    def reduce_decision(self, decision: bool):
        return False

    def broadcast_decision(self, decision: bool):
        return False

    def data_to_device(self, data: Any):
        pass

    def save_checkpoint(self, filepath):
        pass

    def execute_on_rank(self, func: Callable, rank: int):
        pass
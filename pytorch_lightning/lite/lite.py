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
import os
from abc import abstractmethod, ABC
from collections import Callable
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Optional, Sequence, Union, List, Dict, Tuple, Generator

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler, Sampler

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator, TPUAccelerator
from pytorch_lightning.lite.wrappers import _LiteOptimizer, _LiteModule, _LiteDataLoader
from pytorch_lightning.plugins import PLUGIN_INPUT, DDPSpawnPlugin, TrainingTypePlugin, DeepSpeedPlugin
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.utilities import move_data_to_device, DistributedType, DeviceType
from pytorch_lightning.utilities.data import has_iterable_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class LightningLite(ABC):
    def __init__(
        self,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, TrainingTypePlugin]] = None,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        devices: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        precision: Union[int, str] = 32,
    ) -> None:
        self._check_accelerator_support(accelerator)
        self._check_strategy_support(strategy)
        gpu_ids, tpu_cores = Trainer._parse_devices(gpus=gpus, auto_select_gpus=False, tpu_cores=tpu_cores)
        self._accelerator_connector = AcceleratorConnector(
            num_processes=1,
            devices=devices,
            tpu_cores=tpu_cores,
            ipus=None,
            distributed_backend=None,
            accelerator=accelerator,
            strategy=strategy,
            gpus=gpus,
            gpu_ids=gpu_ids,
            num_nodes=num_nodes,
            sync_batchnorm=False,  # TODO: add support?
            benchmark=False,
            replace_sampler_ddp=True,
            deterministic=False,
            precision=precision,
            amp_type="native",
            amp_level=None,
            plugins=plugins,
        )
        self._accelerator = self._accelerator_connector.select_accelerator()
        self._strategy = self._accelerator.training_type_plugin
        self._precision_plugin = self._accelerator.precision_plugin

        # wrap the run method so we can inject setup logic or spawn processes for the user
        setattr(self, "run", self._run_wrapper(self.run))

    @property
    def device(self) -> torch.device:
        return self._accelerator.root_device

    @property
    def global_rank(self) -> int:
        return getattr(self._strategy, "global_rank", 0)

    @property
    def local_rank(self) -> int:
        return getattr(self._strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        return getattr(self._strategy, "node_rank", 0)

    @property
    def world_size(self) -> int:
        return getattr(self._strategy, "world_size", 1)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> None:
        pass

    def setup(
        self,
        models: Union[nn.Module, Sequence[nn.Module]],
        optimizers: Union[Optimizer, Sequence[Optimizer]],
        move_to_device: bool = True,
    ) -> Tuple[Union[nn.Module, Sequence[nn.Module]], Union[Optimizer, Sequence[Optimizer]]]:
        # wrap all objects passed in and return them in the same order
        models = [models] if isinstance(models, nn.Module) else models
        optimizers = [optimizers] if isinstance(optimizers, Optimizer) else optimizers
        models, optimizers = self._setup_models_and_optimizers(models, optimizers)

        if move_to_device:
            models = [self.to_device(model) for model in models]

        models = models[0] if len(models) == 1 else models
        optimizers = optimizers[0] if len(optimizers) == 1 else optimizers
        return models, optimizers

    def setup_dataloaders(
        self, *dataloaders: DataLoader, replace_sampler: bool = True, move_to_device: bool = True
    ) -> Union[DataLoader, Sequence[DataLoader]]:
        # user can call this method independently instead of the general purpose setup method
        dataloaders = [
            self.setup_dataloader(dataloader, replace_sampler=replace_sampler, move_to_device=move_to_device)
            for dataloader in dataloaders
        ]
        dataloaders = dataloaders[0] if len(dataloaders) == 1 else dataloaders
        return dataloaders

    def setup_dataloader(
        self, dataloader: DataLoader, replace_sampler: bool = True, move_to_device: bool = True
    ) -> DataLoader:
        if not replace_sampler or not (
            self._requires_distributed_sampler(dataloader) or isinstance(self._accelerator, TPUAccelerator)
        ):
            return dataloader
        if not isinstance(dataloader.sampler, (SequentialSampler, RandomSampler)):
            raise MisconfigurationException(
                "You seem to have configured a sampler in your DataLoader. This will be replaced "
                " by `DistributedSampler` since `replace_sampler_ddp` is True and you are using"
                " distributed training. Either remove the sampler from your DataLoader or set"
                " `replace_sampler=False` if you want to use your custom sampler."
            )

        sampler = self._get_distributed_sampler(dataloader, **self._strategy.distributed_sampler_kwargs)
        kwargs = TrainerDataLoadingMixin._get_dataloader_init_kwargs(dataloader, sampler)
        device = self.device if move_to_device else None
        return _LiteDataLoader(device=device, **kwargs)

    def backward(self, tensor: Tensor, *args: Any, **kwargs: Any) -> None:
        # user will call self.backward(loss) instead of loss.backward()
        self._accelerator.run_backward(tensor, self._strategy.model, *args, **kwargs)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        with self._accelerator.forward_context():
            yield

    def to_device(self, obj: Union[nn.Module, Tensor, Any]) -> Union[nn.Module, Tensor, Any]:
        if isinstance(obj, nn.Module):
            return obj.to(self.device)
        return move_data_to_device(obj, device=self.device)

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self.local_rank == 0:
            print(*args, **kwargs)

    def reduce_decision(self, decision: bool) -> bool:
        return self._strategy.reduce_boolean_decision(decision)

    def save_checkpoint(self, filepath: Union[str, Path], content: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def execute_on_rank(self, func: Callable, rank: int, *args: Any, **kwargs: Any) -> None:
        if self.global_rank == rank:
            func(*args, **kwargs)

    def _run_wrapper(self, run_method: Callable) -> Callable:
        return partial(self._run_impl, run_method)

    def _run_impl(self, run_method: Callable, *args: Any, **kwargs: Any) -> None:
        self._strategy.setup_environment()
        if isinstance(self._strategy, DDPSpawnPlugin):
            self._strategy.spawn(run_method, *args, **kwargs)
        else:
            run_method(*args, **kwargs)
        # TODO: any teardown needed here?

    def _setup_models_and_optimizers(
        self,
        models: Sequence[nn.Module],
        optimizers: Sequence[Optimizer],
    ) -> Tuple[Sequence[_LiteModule], Sequence[_LiteOptimizer]]:
        # Let accelerator/plugin wrap and connect the models and optimizers
        models, optimizers = self._strategy.setup_models_and_optimizers(models, optimizers)
        models = [_LiteModule(module=model, accelerator=self._accelerator) for model in models]
        optimizers = [_LiteOptimizer(optimizer=optimizer, accelerator=self._accelerator) for optimizer in optimizers]
        return models, optimizers

    def _requires_distributed_sampler(self, dataloader: DataLoader) -> bool:
        return (
            self._accelerator_connector.is_distributed
            and not isinstance(dataloader.sampler, DistributedSampler)
            and not has_iterable_dataset(dataloader)
        )

    @staticmethod
    def _get_distributed_sampler(dataloader: DataLoader, **kwargs: Any) -> DistributedSampler:
        kwargs.setdefault("seed", int(os.getenv("PL_GLOBAL_SEED", 0)))
        sampler = DistributedSampler(dataloader.dataset, **kwargs)
        return sampler

    def _check_accelerator_support(self, accelerator: Optional[Union[str, Accelerator]]) -> None:
        if accelerator is None:
            return
        supported = [t.lower() for t in self._supported_device_types()]
        if not isinstance(accelerator, (Accelerator, str)) or accelerator not in supported:
            raise MisconfigurationException(
                f"`accelerator={repr(accelerator)}` is not a valid choice."
                f" Choose one of {supported} or pass in a `Accelerator` instance."
            )

    def _check_strategy_support(self, strategy: Optional[Union[str, TrainingTypePlugin]]) -> None:
        if strategy is None:
            return
        supported = [t.lower() for t in self._supported_strategy_types()]
        if not isinstance(strategy, (TrainingTypePlugin, str)) or strategy not in supported:
            raise MisconfigurationException(
                f"`strategy={repr(strategy)}` is not a valid choice."
                f" Choose one of {supported} or pass in a `TrainingTypePlugin` instance."
            )

    @staticmethod
    def _supported_device_types() -> Sequence[DeviceType]:
        return (
            DeviceType.CPU,
            DeviceType.GPU,
            DeviceType.TPU,
        )

    @staticmethod
    def _supported_strategy_types() -> Sequence[str]:
        return (
            DistributedType.DP,
            DistributedType.DDP,
            DistributedType.DDP_SPAWN,
            DistributedType.TPU_SPAWN,
            DistributedType.DP,
            DistributedType.DEEPSPEED,
            DistributedType.DDP_SHARDED,
        )

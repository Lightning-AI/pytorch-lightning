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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast, Dict, Generator, List, Optional, overload, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.lite.wrappers import _LiteDataLoader, _LiteModule, _LiteOptimizer
from pytorch_lightning.plugins import PLUGIN_INPUT
from pytorch_lightning.strategies import DeepSpeedStrategy, Strategy, TPUSpawnStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.utilities import _AcceleratorType, _StrategyType, move_data_to_device
from pytorch_lightning.utilities.apply_func import apply_to_collection, convert_to_tensors
from pytorch_lightning.utilities.data import (
    _auto_add_worker_init_fn,
    _replace_dataloader_init_method,
    _update_dataloader,
    has_iterable_dataset,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import seed_everything


class LightningLite(ABC):
    """Lite accelerates your PyTorch training or inference code with minimal changes required.

    - Automatic placement of models and data onto the device.
    - Automatic support for mixed and double precision (smaller memory footprint).
    - Seamless switching between hardware (CPU, GPU, TPU) and distributed training strategies
      (data-parallel training, sharded training, etc.).
    - Automated spawning of processes, no launch utilities required.
    - Multi-node support.

    Args:
        accelerator: The hardware to run on. Possible choices are: ``"cpu"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        strategy: Strategy for how to run across multiple devices. Possible choices are:
            ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"ddp_sharded"``.
        devices: Number of devices to train on (``int``), which GPUs to train on (``list`` or ``str``), or ``"auto"``.
            The value applies per node.
        num_nodes: Number of GPU nodes for distributed training.
        precision: Double precision (``64``), full precision (``32``), half precision (``16``),
            or bfloat16 precision (``"bf16"``).
        plugins: One or several custom plugins
        gpus: Provides the same function as the ``devices`` argument but implies ``accelerator="gpu"``.
        tpu_cores: Provides the same function as the ``devices`` argument but implies ``accelerator="tpu"``.
    """

    def __init__(
        self,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        devices: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        precision: Union[int, str] = 32,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
    ) -> None:
        self._check_accelerator_support(accelerator)
        self._check_strategy_support(strategy)
        self._accelerator_connector = AcceleratorConnector(
            num_processes=None,
            devices=devices,
            tpu_cores=tpu_cores,
            ipus=None,
            accelerator=accelerator,
            strategy=strategy,
            gpus=gpus,
            num_nodes=num_nodes,
            sync_batchnorm=False,  # TODO: add support?
            benchmark=False,
            replace_sampler_ddp=True,
            deterministic=False,
            precision=precision,
            amp_type="native",
            amp_level=None,
            plugins=plugins,
            auto_select_gpus=False,
        )
        self._strategy = self._accelerator_connector.strategy
        self._accelerator = self._strategy.accelerator
        self._precision_plugin = self._strategy.precision_plugin
        self._models_setup: int = 0

        # wrap the run method so we can inject setup logic or spawn processes for the user
        setattr(self, "run", partial(self._run_impl, self.run))

    @property
    def device(self) -> torch.device:
        """The current device this process runs on.

        Use this to create tensors directly on the device if needed.
        """
        return self._strategy.root_device

    @property
    def global_rank(self) -> int:
        """The global index of the current process across all devices and nodes."""
        return getattr(self._strategy, "global_rank", 0)

    @property
    def local_rank(self) -> int:
        """The index of the current process among the processes running on the local node."""
        return getattr(self._strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        """The index of the current node."""
        return getattr(self._strategy, "node_rank", 0)

    @property
    def world_size(self) -> int:
        """The total number of processes running across all devices and nodes."""
        return getattr(self._strategy, "world_size", 1)

    @property
    def is_global_zero(self) -> bool:
        """Wether this rank is rank zero."""
        return self._strategy.is_global_zero

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """All the code inside this run method gets accelerated by Lite.

        You can pass arbitrary arguments to this function when overriding it.
        """

    def setup(
        self,
        model: nn.Module,
        *optimizers: Optimizer,
        move_to_device: bool = True,
    ) -> Any:  # no specific return because the way we want our API to look does not play well with mypy
        """Setup a model and its optimizers for accelerated training.

        Args:
            model: A model to setup
            *optimizers: The optimizer(s) to setup (no optimizers is also possible)
            move_to_device: If set ``True`` (default), moves the model to the correct device. Set this to ``False``
                and alternatively use :meth:`to_device` manually.

        Returns:
            The tuple of the wrapped model and list of optimizers, in the same order they were passed in.
        """
        self._validate_setup(model, optimizers)

        if move_to_device:
            model = self._move_model_to_device(model=model, optimizers=list(optimizers))

        # Let accelerator/plugin wrap and connect the models and optimizers
        model, optimizers = self._strategy._setup_model_and_optimizers(model, list(optimizers))
        model = _LiteModule(model, self._precision_plugin)
        optimizers = [_LiteOptimizer(optimizer=optimizer, strategy=self._strategy) for optimizer in optimizers]
        self._models_setup += 1
        if optimizers:
            # join both types in a list for API convenience
            return [model] + optimizers  # type: ignore
        return model

    def setup_dataloaders(
        self, *dataloaders: DataLoader, replace_sampler: bool = True, move_to_device: bool = True
    ) -> Union[DataLoader, List[DataLoader]]:
        """Setup one or multiple dataloaders for accelerated training. If you need different settings for each
        dataloader, call this method individually for each one.

        Args:
            *dataloaders: A single dataloader or a sequence of dataloaders.
            replace_sampler: If set ``True`` (default), automatically wraps or replaces the sampler on the dataloader(s)
                for distributed training. If you have a custom sampler defined, set this to this argument to ``False``.
            move_to_device: If set ``True`` (default), moves the data returned by the dataloader(s) automatically to
                the correct device. Set this to ``False`` and alternatively use :meth:`to_device` manually on the
                returned data.

        Returns:
            The wrapped dataloaders, in the same order they were passed in.
        """
        self._validate_setup_dataloaders(dataloaders)
        dataloaders = [
            self._setup_dataloader(dataloader, replace_sampler=replace_sampler, move_to_device=move_to_device)
            for dataloader in dataloaders
        ]
        dataloaders = dataloaders[0] if len(dataloaders) == 1 else dataloaders
        return dataloaders  # type: ignore[return-value]

    def _setup_dataloader(
        self, dataloader: DataLoader, replace_sampler: bool = True, move_to_device: bool = True
    ) -> DataLoader:
        """Setup a single dataloader for accelerated training.

        Args:
            dataloader: The dataloader to accelerate.
            replace_sampler: If set ``True`` (default), automatically wraps or replaces the sampler on the dataloader
                for distributed training. If you have a custom sampler defined, set this to this argument to ``False``.
            move_to_device: If set ``True`` (default), moves the data returned by the dataloader automatically to
                the correct device. Set this to ``False`` and alternatively use :meth:`to_device` manually on the
                returned data.

        Returns:
            The wrapped dataloader.
        """
        sampler = dataloader.sampler
        if replace_sampler and self._requires_distributed_sampler(dataloader):
            if not isinstance(sampler, (SequentialSampler, RandomSampler)):
                raise MisconfigurationException(
                    "You seem to have configured a sampler in your DataLoader. This will be replaced "
                    " by `DistributedSampler` since `replace_sampler_ddp` is True and you are using"
                    " distributed training. Either remove the sampler from your DataLoader or set"
                    " `replace_sampler=False` if you want to use your custom sampler."
                )
            sampler = self._get_distributed_sampler(dataloader, **self._strategy.distributed_sampler_kwargs)

        # the dataloader needs to be re-instantiated because we want to update the input arguments (e.g., sampler)
        dataloader = _update_dataloader(dataloader, sampler)

        # add worker_init_fn for correct seeding in worker processes
        _auto_add_worker_init_fn(dataloader, self.global_rank)

        dataloader = self._strategy.process_dataloader(dataloader)
        device = self.device if move_to_device and not isinstance(self._strategy, TPUSpawnStrategy) else None
        lite_dataloader = _LiteDataLoader(dataloader=dataloader, device=device)
        lite_dataloader = cast(DataLoader, lite_dataloader)
        return lite_dataloader

    def backward(self, tensor: Tensor, *args: Any, model: Optional[_LiteModule] = None, **kwargs: Any) -> None:
        """Replaces ``loss.backward()`` in your training loop. Handles precision and automatically for you.

        Args:
            tensor: The tensor (loss) to back-propagate gradients from.
            *args: Optional positional arguments passed to the underlying backward function.
            model: Optional model instance for plugins that require the model for backward().
            **kwargs: Optional named keyword arguments passed to the underlying backward function.

        Note:
            When using ``strategy="deepspeed"`` and multiple models were setup, it is required to pass in the
            model as argument here.
        """
        module = model.module if model is not None else model
        if isinstance(self._strategy, DeepSpeedStrategy):
            if model is None:
                if self._models_setup == 0:
                    raise MisconfigurationException(
                        "No models were setup for backward. Did you forget to call `self.setup()`?"
                    )
                if self._models_setup > 1:
                    raise MisconfigurationException(
                        "When using multiple models + deepspeed, please provide the model used to perform"
                        " the optimization: `self.backward(loss, model=model)`"
                    )
                module = self._strategy.model
            else:
                # requires to attach the current `DeepSpeedEngine` for the `_LiteOptimizer.step` call.
                self._strategy.model = module

        self._precision_plugin._run_backward(tensor, module, *args, **kwargs)

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """A context manager to automatically convert operations for the chosen precision.

        Use this only if the `forward` method of your model does not cover all operations you wish to run with the
        chosen precision setting.
        """
        with self._precision_plugin.forward_context():
            yield

    @overload
    def to_device(self, obj: nn.Module) -> nn.Module:
        ...

    @overload
    def to_device(self, obj: Tensor) -> Tensor:
        ...

    @overload
    def to_device(self, obj: Any) -> Any:
        ...

    def to_device(self, obj: Union[nn.Module, Tensor, Any]) -> Union[nn.Module, Tensor, Any]:
        """Move a :class:`torch.nn.Module` or a collection of tensors to the current device, if it is not already
        on that device.

        Args:
            obj: An object to move to the device. Can be an instance of :class:`torch.nn.Module`, a tensor, or a
                 (nested) collection of tensors (e.g., a dictionary).

        Returns:
            A reference to the object that was moved to the new device.
        """
        if isinstance(obj, nn.Module):
            if self.device.type == "cuda":
                # need to call this manually here again in case we spawned with DDPSpawnStrategy
                # TODO: refactor to let plugin handle this cleanly
                torch.cuda.set_device(self.device)
            return obj.to(self.device)
        return move_data_to_device(obj, device=self.device)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print something only on the first process.

        Arguments passed to this method are forwarded to the Python built-in :func:`print` function.
        """
        if self.local_rank == 0:
            print(*args, **kwargs)

    def barrier(self, name: Optional[str] = None) -> None:
        """Wait for all processes to enter this call. Use this to synchronize all parallel processes, but only if
        necessary, otherwise the overhead of synchronization will cause your program to slow down.

        Example::

            if self.global_rank == 0:
                # let process 0 download the dataset
                dataset.download_files()

            # let all processes wait before reading the dataset
            self.barrier()

            # now all processes can read the files and start training
        """
        self._strategy.barrier(name=name)

    def all_gather(
        self, data: Union[torch.Tensor, Dict, List, Tuple], group: Optional[Any] = None, sync_grads: bool = False
    ) -> Union[torch.Tensor, Dict, List, Tuple]:
        r"""
        Gather tensors or collections of tensors from multiple processes.

        Args:
            data: int, float, tensor of shape (batch, ...), or a (possibly nested) collection thereof.
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for the all_gather operation

        Return:
            A tensor of shape (world_size, batch, ...), or if the input was a collection
            the output will also be a collection with tensors of this shape.
        """
        group = group if group is not None else torch.distributed.group.WORLD
        data = convert_to_tensors(data, device=self.device)
        return apply_to_collection(data, torch.Tensor, self._strategy.all_gather, group=group, sync_grads=sync_grads)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return self._strategy.broadcast(obj, src=src)

    def save(self, content: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save checkpoint contents to a file.

        How and which processes save gets determined by the `strategy`. For example, the `ddp` strategy
        saves checkpoints only on process 0.

        Args:
            content: A dictionary with contents, i.e., the state dict of your model
            filepath: A path to where the file should be saved
        """
        self._strategy.save_checkpoint(content, filepath)

    def load(self, filepath: Union[str, Path]) -> Any:
        """Load a checkpoint from a file.

        How and which processes load gets determined by the `strategy`

        Args:
            filepath: A path to where the file is located
        """
        return self._strategy.load_checkpoint(filepath)

    @staticmethod
    def seed_everything(seed: Optional[int] = None, workers: Optional[bool] = None) -> int:
        """Helper function to seed everything without explicitly importing Lightning.

        See :func:`pytorch_lightning.seed_everything` for more details.
        """
        if workers is None:
            # Lightning sets `workers=False` by default to avoid breaking reproducibility, but since this is a new
            # release, we can afford to do it.
            workers = True
        return seed_everything(seed=seed, workers=workers)

    def _run_impl(self, run_method: Callable, *args: Any, **kwargs: Any) -> Any:
        # apply sharded context to prevent OOM
        run_method = partial(self._run_with_strategy_setup, run_method)

        if self._strategy.launcher is not None:
            return self._strategy.launcher.launch(run_method, *args, **kwargs)
        else:
            return run_method(*args, **kwargs)

    def _run_with_strategy_setup(self, run_method: Callable, *args: Any, **kwargs: Any) -> Any:
        self._strategy.setup_environment()
        with self._strategy.model_sharded_context(), _replace_dataloader_init_method():
            return run_method(*args, **kwargs)

    def _move_model_to_device(self, model: nn.Module, optimizers: List[Optimizer]) -> nn.Module:
        if isinstance(self._strategy, TPUSpawnStrategy):
            # When the user creates the optimizer, they reference the parameters on the CPU.
            # However, when running with TPU the parameters get copied and the reference in the optimizer
            # remains invalid. We need to update the references to point to the parameter tensors on the device.
            params_before_move = dict(model.named_parameters())
            model = self.to_device(model)
            # XLA makes a copy on the parameters, so the device is not the same before and after to_device.
            params_on_device = dict(model.named_parameters())

            mapping = {param: params_on_device[name] for name, param in params_before_move.items()}
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group["params"] = [mapping.get(p, p) for p in param_group["params"]]
        else:
            model = self.to_device(model)
        return model

    def _requires_distributed_sampler(self, dataloader: DataLoader) -> bool:
        return (
            self._accelerator_connector.is_distributed
            and not isinstance(dataloader.sampler, DistributedSampler)
            and not has_iterable_dataset(dataloader)
        )

    @staticmethod
    def _get_distributed_sampler(dataloader: DataLoader, **kwargs: Any) -> DistributedSampler:
        kwargs.setdefault("seed", int(os.getenv("PL_GLOBAL_SEED", 0)))
        return DistributedSampler(dataloader.dataset, **kwargs)

    def _check_accelerator_support(self, accelerator: Optional[Union[str, Accelerator]]) -> None:
        supported = [t.value.lower() for t in self._supported_device_types()] + ["auto"]
        valid = accelerator is None or isinstance(accelerator, Accelerator) or accelerator in supported
        if not valid:
            raise MisconfigurationException(
                f"`accelerator={repr(accelerator)}` is not a valid choice."
                f" Choose one of {supported} or pass in a `Accelerator` instance."
            )

    def _check_strategy_support(self, strategy: Optional[Union[str, Strategy]]) -> None:
        supported = [t.lower() for t in self._supported_strategy_types()]
        valid = strategy is None or isinstance(strategy, Strategy) or strategy in supported
        if not valid:
            raise MisconfigurationException(
                f"`strategy={repr(strategy)}` is not a valid choice."
                f" Choose one of {supported} or pass in a `Strategy` instance."
            )

    @staticmethod
    def _supported_device_types() -> Sequence[_AcceleratorType]:
        return (
            _AcceleratorType.CPU,
            _AcceleratorType.GPU,
            _AcceleratorType.TPU,
        )

    @staticmethod
    def _supported_strategy_types() -> Sequence[_StrategyType]:
        return (
            _StrategyType.DP,
            _StrategyType.DDP,
            _StrategyType.DDP_SPAWN,
            _StrategyType.DEEPSPEED,
            _StrategyType.DDP_SHARDED,
            _StrategyType.DDP_SHARDED_SPAWN,
        )

    @staticmethod
    def _validate_setup(model: nn.Module, optimizers: Sequence[Optimizer]) -> None:
        if isinstance(model, _LiteModule):
            raise MisconfigurationException("A model should be passed only once to the `setup` method.")

        if any(isinstance(opt, _LiteOptimizer) for opt in optimizers):
            raise MisconfigurationException("An optimizer should be passed only once to the `setup` method.")

    @staticmethod
    def _validate_setup_dataloaders(dataloaders: Sequence[DataLoader]) -> None:
        if any(isinstance(dl, _LiteDataLoader) for dl in dataloaders):
            raise MisconfigurationException("A dataloader should be passed only once to the `setup_dataloaders` method")

        if any(not isinstance(dl, DataLoader) for dl in dataloaders):
            raise MisconfigurationException("Only PyTorch DataLoader are currently supported in `setup_dataloaders`.")

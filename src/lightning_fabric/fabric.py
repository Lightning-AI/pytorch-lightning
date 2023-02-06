# Copyright The Lightning AI team.
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
import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast, Dict, Generator, List, Mapping, Optional, overload, Sequence, Tuple, Union

import torch
import torch.nn as nn
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.overrides import is_overridden
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from lightning_fabric.loggers import Logger

from lightning_fabric.plugins import Precision  # avoid circular imports: # isort: split
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.connector import _Connector, _PLUGIN_INPUT, _PRECISION_INPUT
from lightning_fabric.strategies import DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy, Strategy, XLAStrategy
from lightning_fabric.strategies.strategy import _Sharded, TBroadcast
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors
from lightning_fabric.utilities.data import (
    _auto_add_worker_init_fn,
    _replace_dunder_methods,
    _update_dataloader,
    has_iterable_dataset,
)
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from lightning_fabric.utilities.seed import seed_everything
from lightning_fabric.utilities.warnings import PossibleUserWarning
from lightning_fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer


class Fabric:
    """Fabric accelerates your PyTorch training or inference code with minimal changes required.

    - Automatic placement of models and data onto the device.
    - Automatic support for mixed and double precision (smaller memory footprint).
    - Seamless switching between hardware (CPU, GPU, TPU) and distributed training strategies
      (data-parallel training, sharded training, etc.).
    - Automated spawning of processes, no launch utilities required.
    - Multi-node support.

    Args:
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        strategy: Strategy for how to run across multiple devices. Possible choices are:
            ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
        devices: Number of devices to train on (``int``), which GPUs to train on (``list`` or ``str``), or ``"auto"``.
            The value applies per node.
        num_nodes: Number of GPU nodes for distributed training.
        precision: Double precision (``64``), full precision (``32``), half precision (``16``),
            or bfloat16 precision (``"bf16"``).
        plugins: One or several custom plugins
        callbacks: A single callback or a list of callbacks. A callback can contain any arbitrary methods that
            can be invoked through :meth:`~lightning_fabric.fabric.Fabric.call` by the user.
        loggers: A single logger or a list of loggers. See :meth:`~lightning_fabric.fabric.Fabric.log` for more
            information.
    """

    def __init__(
        self,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        devices: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        precision: _PRECISION_INPUT = 32,
        plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
    ) -> None:
        self._connector = _Connector(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
        )
        self._strategy: Strategy = self._connector.strategy
        self._accelerator: Accelerator = self._connector.accelerator
        self._precision: Precision = self._strategy.precision
        callbacks = callbacks if callbacks is not None else []
        self._callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
        loggers = loggers if loggers is not None else []
        self._loggers = loggers if isinstance(loggers, list) else [loggers]
        self._models_setup: int = 0

        self._prepare_run_method()
        if _is_using_cli():
            # when the CLI is used to launch the script, we need to set up the environment (init processes) here so
            # that the user can immediately use all functionality in strategies
            self._strategy.setup_environment()

    @property
    def accelerator(self) -> Accelerator:
        return self._accelerator

    @property
    def strategy(self) -> Strategy:
        return self._strategy

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
        """Whether this rank is rank zero."""
        return self._strategy.is_global_zero

    @property
    def loggers(self) -> List[Logger]:
        """Returns all loggers passed to Fabric."""
        return self._loggers

    @property
    def logger(self) -> Logger:
        """Returns the first logger in the list passed to Fabric, which is considered the main logger."""
        return self._loggers[0]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """All the code inside this run method gets accelerated by Fabric.

        You can pass arbitrary arguments to this function when overriding it.
        """

    def setup(
        self,
        module: nn.Module,
        *optimizers: Optimizer,
        move_to_device: bool = True,
    ) -> Any:  # no specific return because the way we want our API to look does not play well with mypy
        """Set up a model and its optimizers for accelerated training.

        Args:
            module: A :class:`torch.nn.Module` to set up
            *optimizers: The optimizer(s) to set up (no optimizers is also possible)
            move_to_device: If set ``True`` (default), moves the model to the correct device. Set this to ``False``
                and alternatively use :meth:`to_device` manually.

        Returns:
            The tuple containing wrapped module and the optimizers, in the same order they were passed in.
        """
        self._validate_setup(module, optimizers)
        original_module = module

        module = self._precision.convert_module(module)

        if move_to_device:
            module = self._move_model_to_device(model=module, optimizers=list(optimizers))

        # Let accelerator/plugin wrap and connect the models and optimizers
        if optimizers:
            module, optimizers = self._strategy.setup_module_and_optimizers(  # type: ignore[assignment]
                module, list(optimizers)
            )
        else:
            module = self._strategy.setup_module(module)

        module = _FabricModule(module, self._precision, original_module=original_module)

        # Update the _DeviceDtypeModuleMixin's device parameter
        module.to(self.device if move_to_device else next(module.parameters()).device)

        optimizers = [_FabricOptimizer(optimizer=optimizer, strategy=self._strategy) for optimizer in optimizers]

        self._models_setup += 1

        if hasattr(original_module, "_fabric"):  # this is probably a LightningModule
            original_module._fabric = self  # type: ignore[assignment]
            original_module._fabric_optimizers = optimizers  # type: ignore[assignment]

        if optimizers:
            # join both types in a tuple for API convenience
            return tuple((module, *optimizers))
        return module

    def setup_module(self, module: nn.Module, move_to_device: bool = True) -> _FabricModule:
        """Set up a model for accelerated training or inference.

        This is the same as calling ``.setup(model)`` with no optimizers. It is useful for inference or for certain
        strategies like `FSDP` that require setting up the module before the optimizer can be created and set up.
        See also :meth:`setup_optimizers`.

        Args:
            module: A :class:`torch.nn.Module` to set up
            move_to_device: If set ``True`` (default), moves the model to the correct device. Set this to ``False``
                and alternatively use :meth:`to_device` manually.

        Returns:
            The wrapped model.
        """
        self._validate_setup_module(module)
        original_module = module

        module = self._precision.convert_module(module)

        if move_to_device:
            module = self._move_model_to_device(model=module, optimizers=[])

        # Let strategy wrap and connect the module alone
        module = self._strategy.setup_module(module)
        module = _FabricModule(module, self._precision, original_module=original_module)

        if not isinstance(self._strategy, FSDPStrategy):
            # Update the _DeviceDtypeModuleMixin's device parameter
            module.to(self.device if move_to_device else next(module.parameters()).device)

        if hasattr(original_module, "_fabric"):  # this is probably a LightningModule
            original_module._fabric = self  # type: ignore[assignment]

        self._models_setup += 1
        return module

    def setup_optimizers(self, *optimizers: Optimizer) -> Union[_FabricOptimizer, Tuple[_FabricOptimizer, ...]]:
        """Set up one or more optimizers for accelerated training.

        Some strategies do not allow setting up model and optimizer independently. For them, you should call
        ``.setup(model, optimizer, ...)`` instead to jointly set them up.

        Args:
            *optimizers: One or more optmizers to set up.

        Returns:
            The wrapped optimizer(s).
        """
        self._validate_setup_optimizers(optimizers)
        optimizers = [self._strategy.setup_optimizer(optimizer) for optimizer in optimizers]
        optimizers = [_FabricOptimizer(optimizer=optimizer, strategy=self._strategy) for optimizer in optimizers]
        return optimizers[0] if len(optimizers) == 1 else tuple(optimizers)

    def setup_dataloaders(
        self, *dataloaders: DataLoader, replace_sampler: bool = True, move_to_device: bool = True
    ) -> Union[DataLoader, List[DataLoader]]:
        """Set up one or multiple dataloaders for accelerated training. If you need different settings for each
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
        """Set up a single dataloader for accelerated training.

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
            sampler = self._get_distributed_sampler(dataloader, **self._strategy.distributed_sampler_kwargs)

        # the dataloader needs to be re-instantiated because we want to update the input arguments (e.g., sampler)
        dataloader = _update_dataloader(dataloader, sampler)

        # add worker_init_fn for correct seeding in worker processes
        _auto_add_worker_init_fn(dataloader, self.global_rank)

        dataloader = self._strategy.process_dataloader(dataloader)
        device = self.device if move_to_device and not isinstance(self._strategy, XLAStrategy) else None
        lite_dataloader = _FabricDataLoader(dataloader=dataloader, device=device)
        lite_dataloader = cast(DataLoader, lite_dataloader)
        return lite_dataloader

    def backward(self, tensor: Tensor, *args: Any, model: Optional[_FabricModule] = None, **kwargs: Any) -> None:
        """Replaces ``loss.backward()`` in your training loop. Handles precision and automatically for you.

        Args:
            tensor: The tensor (loss) to back-propagate gradients from.
            *args: Optional positional arguments passed to the underlying backward function.
            model: Optional model instance for plugins that require the model for backward().
            **kwargs: Optional named keyword arguments passed to the underlying backward function.

        Note:
            When using ``strategy="deepspeed"`` and multiple models were set up, it is required to pass in the
            model as argument here.
        """
        module = model._forward_module if model is not None else model
        if isinstance(self._strategy, DeepSpeedStrategy):
            if model is None:
                if self._models_setup == 0:
                    raise RuntimeError("No models were set up for backward. Did you forget to call `self.setup()`?")
                if self._models_setup > 1:
                    raise ValueError(
                        "When using multiple models + deepspeed, please provide the model used to perform"
                        " the optimization: `self.backward(loss, model=model)`"
                    )
                module = self._strategy.model
            else:
                # requires to attach the current `DeepSpeedEngine` for the `_FabricOptimizer.step` call.
                self._strategy._deepspeed_engine = module

        self._precision.backward(tensor, module, *args, **kwargs)

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """A context manager to automatically convert operations for the chosen precision.

        Use this only if the `forward` method of your model does not cover all operations you wish to run with the
        chosen precision setting.
        """
        with self._precision.forward_context():
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
            self._accelerator.setup_device(self.device)
            self._strategy.module_to_device(obj)
            return obj
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
        self, data: Union[Tensor, Dict, List, Tuple], group: Optional[Any] = None, sync_grads: bool = False
    ) -> Union[Tensor, Dict, List, Tuple]:
        r"""Gather tensors or collections of tensors from multiple processes.

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
        return apply_to_collection(data, Tensor, self._strategy.all_gather, group=group, sync_grads=sync_grads)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return self._strategy.broadcast(obj, src=src)

    @contextmanager
    def no_backward_sync(self, module: _FabricModule, enabled: bool = True) -> Generator:
        """Skip gradient synchronization during backward to avoid redundant communication overhead.

        Use this context manager when performing gradient accumulation to speed up training with multiple devices.

        Example::

            # Accumulate gradient 8 batches at a time
            with self.no_backward_sync(model, enabled=(batch_idx % 8 != 0)):
                output = model(input)
                loss = ...
                self.backward(loss)
                ...

        For those strategies that don't support it, a warning is emitted. For single-device strategies, it is a no-op.
        Both the model's `.forward()` and the `self.backward()` call need to run under this context.

        Args:
            module: The module for which to control the gradient synchronization.
            enabled: Whether the context manager is enabled or not. ``True`` means skip the sync, ``False`` means do not
                skip.
        """

        if not isinstance(module, _FabricModule):
            raise TypeError(
                "You need to set up the model first before you can call `self.no_backward_sync()`:"
                " `model = self.setup(model, ...)`"
            )
        if not enabled or isinstance(self._strategy, SingleDeviceStrategy):
            context = nullcontext()
        elif self._strategy._backward_sync_control is None:
            rank_zero_warn(
                f"The `{self._strategy.__class__.__name__}` does not support skipping the gradient synchronization."
                f" Remove `.no_backward_sync()` from your code or choose a different strategy.",
                category=PossibleUserWarning,
            )
            context = nullcontext()
        else:
            context = self._strategy._backward_sync_control.no_backward_sync(  # type: ignore[assignment]
                module._forward_module
            )

        with context:
            yield

    @contextmanager
    def sharded_model(self) -> Generator:
        """Shard the parameters of the model instantly when instantiating the layers.

        Use this context manager with strategies that support sharding the model parameters to save peak memory usage.

        Example::

            with self.sharded_model():
                model = MyModel()

        The context manager is strategy-agnostic and for the ones that don't do sharding, it is a no-op.
        """
        if isinstance(self._strategy, _Sharded):
            with self._strategy.module_sharded_context():
                yield
        else:
            yield

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

    def launch(self, function: Optional[Callable[["Fabric"], Any]] = None, *args: Any, **kwargs: Any) -> Any:
        if _is_using_cli():
            raise RuntimeError(
                "This script was launched through the CLI, and processes have already been created. Calling "
                " `.launch()` again is not allowed."
            )
        if function is not None and not inspect.signature(function).parameters:
            raise TypeError(
                "The function passed to `Fabric.launch()` needs to take at least one argument. The launcher will pass"
                " in the `Fabric` object so you can use it inside the function."
            )
        function = partial(self._run_with_setup, function or _do_nothing)
        args = [self, *args]
        if self._strategy.launcher is not None:
            return self._strategy.launcher.launch(function, *args, **kwargs)
        return function(*args, **kwargs)

    def call(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Trigger the callback methods with the given name and arguments.

        Not all objects registered via ``Fabric(callbacks=...)`` must implement a method with the given name. The ones
        that have a matching method name will get called.

        Args:
            hook_name: The name of the callback method.
            *args: Optional positional arguments that get passed down to the callback method.
            **kwargs: Optional keyword arguments that get passed down to the callback method.

        Example::

            class MyCallback:
                def on_train_epoch_end(self, results):
                    ...

            fabric = Fabric(callbacks=[MyCallback()])
            fabric.call("on_train_epoch_end", results={...})
        """
        for callback in self._callbacks:
            method = getattr(callback, hook_name, None)
            if method is None:
                continue
            if not callable(method):
                rank_zero_warn(
                    f"Skipping the callback `{type(callback).__name__}.{hook_name}` because it is not callable."
                )
                continue

            method(*args, **kwargs)

            # TODO(fabric): handle the following signatures
            # method(self, fabric|trainer, x, y=1)
            # method(self, fabric|trainer, *args, x, y=1)
            # method(self, *args, y=1)
            # method(self, *args, **kwargs)

    def log(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Log a scalar to all loggers that were added to Fabric.

        Args:
            name: The name of the metric to log.
            value: The metric value to collect. If the value is a :class:`torch.Tensor`, it gets detached from the
                graph automatically.
            step: Optional step number. Most Logger implementations auto-increment the step value by one with every
                log call. You can specify your own value here.
        """
        self.log_dict(metrics={name: value}, step=step)

    def log_dict(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Log multiple scalars at once to all loggers that were added to Fabric.

        Args:
            metrics: A dictionary where the key is the name of the metric and the value the scalar to be logged.
                Any :class:`torch.Tensor` in the dictionary get detached from the graph automatically.
            step: Optional step number. Most Logger implementations auto-increment this value by one with every
                log call. You can specify your own value here.
        """
        metrics = convert_tensors_to_scalars(metrics)
        for logger in self._loggers:
            logger.log_metrics(metrics=metrics, step=step)

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
        run_method = partial(self._run_with_setup, run_method)
        if self._strategy.launcher is not None:
            return self._strategy.launcher.launch(run_method, *args, **kwargs)
        else:
            return run_method(*args, **kwargs)

    def _run_with_setup(self, run_function: Callable, *args: Any, **kwargs: Any) -> Any:
        self._strategy.setup_environment()
        # apply sharded context to prevent OOM
        with self.sharded_model(), _replace_dunder_methods(DataLoader, "dataset"), _replace_dunder_methods(
            BatchSampler
        ):
            return run_function(*args, **kwargs)

    def _move_model_to_device(self, model: nn.Module, optimizers: List[Optimizer]) -> nn.Module:
        initial_device = next(model.parameters()).device
        if any(param.device != initial_device for param in model.parameters()):
            rank_zero_warn(
                "The model passed to `Fabric.setup()` has parameters on different devices. Since `move_to_device=True`,"
                " all parameters will be moved to the new device. If this is not desired, set "
                " `Fabric.setup(..., move_to_device=False)`.",
                category=PossibleUserWarning,
            )

        if isinstance(self._strategy, XLAStrategy):
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
            getattr(self.strategy, "distributed_sampler_kwargs", None) is not None
            and not isinstance(dataloader.sampler, DistributedSampler)
            and not has_iterable_dataset(dataloader)
        )

    @staticmethod
    def _get_distributed_sampler(dataloader: DataLoader, **kwargs: Any) -> DistributedSampler:
        kwargs.setdefault("shuffle", isinstance(dataloader.sampler, RandomSampler))
        kwargs.setdefault("seed", int(os.getenv("PL_GLOBAL_SEED", 0)))
        if isinstance(dataloader.sampler, (RandomSampler, SequentialSampler)):
            return DistributedSampler(dataloader.dataset, **kwargs)
        return DistributedSamplerWrapper(dataloader.sampler, **kwargs)

    def _prepare_run_method(self) -> None:
        if is_overridden("run", self, Fabric) and _is_using_cli():
            raise TypeError(
                "Overriding `Fabric.run()` and launching from the CLI is not allowed. Run the script normally,"
                " or change your code to directly call `fabric = Fabric(...); fabric.setup(...)` etc."
            )
        # wrap the run method, so we can inject setup logic or spawn processes for the user
        setattr(self, "run", partial(self._run_impl, self.run))

    def _validate_setup(self, module: nn.Module, optimizers: Sequence[Optimizer]) -> None:
        if isinstance(module, _FabricModule):
            raise ValueError("A model should be passed only once to the `setup` method.")

        if any(isinstance(opt, _FabricOptimizer) for opt in optimizers):
            raise ValueError("An optimizer should be passed only once to the `setup` method.")

        if isinstance(self._strategy, FSDPStrategy):
            raise RuntimeError(
                f"The `{type(self).__name__}` requires the model and optimizer(s) to be set up separately."
                " Create and set up the model first through `model = self.setup_model(model)`. Then create the"
                " optimizer and set it up: `optimizer = self.setup_optimizer(optimizer)`."
            )

    def _validate_setup_module(self, module: nn.Module) -> None:
        if isinstance(module, _FabricModule):
            raise ValueError("A model should be passed only once to the `setup_module` method.")

    def _validate_setup_optimizers(self, optimizers: Sequence[Optimizer]) -> None:
        if isinstance(self._strategy, (DeepSpeedStrategy, XLAStrategy)):
            raise RuntimeError(
                f"The `{type(self._strategy).__name__}` requires the model and optimizer(s) to be set up jointly"
                " through `.setup(model, optimizer, ...)`."
            )

        if not optimizers:
            raise ValueError("`setup_optimizers` requires at least one optimizer as input.")

        if any(isinstance(opt, _FabricOptimizer) for opt in optimizers):
            raise ValueError("An optimizer should be passed only once to the `setup_optimizers` method.")

    @staticmethod
    def _validate_setup_dataloaders(dataloaders: Sequence[DataLoader]) -> None:
        if not dataloaders:
            raise ValueError("`setup_dataloaders` requires at least one dataloader as input.")

        if any(isinstance(dl, _FabricDataLoader) for dl in dataloaders):
            raise ValueError("A dataloader should be passed only once to the `setup_dataloaders` method.")

        if any(not isinstance(dl, DataLoader) for dl in dataloaders):
            raise TypeError("Only PyTorch DataLoader are currently supported in `setup_dataloaders`.")


def _is_using_cli() -> bool:
    return bool(int(os.environ.get("LT_CLI_USED", "0")))


def _do_nothing(*_: Any) -> None:
    pass

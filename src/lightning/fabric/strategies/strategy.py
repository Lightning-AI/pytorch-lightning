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
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.launchers.launcher import _Launcher
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.fabric.utilities.types import _PATH, _Stateful, Optimizable, ReduceOp

TBroadcast = TypeVar("TBroadcast")
TReduce = TypeVar("TReduce")

log = logging.getLogger(__name__)


class Strategy(ABC):
    """Base class for all strategies that change the behaviour of the training, validation and test- loop."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
    ) -> None:
        self._accelerator: Optional[Accelerator] = accelerator
        self._checkpoint_io: Optional[CheckpointIO] = checkpoint_io
        self._precision: Optional[Precision] = precision
        self._launcher: Optional[_Launcher] = None
        self._backward_sync_control: Optional[_BackwardSyncControl] = None

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        """Returns the root device."""

    @property
    @abstractmethod
    def is_global_zero(self) -> bool:
        """Whether the current process is the rank zero process not only on the local node, but for all nodes."""

    @property
    def launcher(self) -> Optional[_Launcher]:
        return self._launcher

    @property
    def accelerator(self) -> Optional[Accelerator]:
        return self._accelerator

    @accelerator.setter
    def accelerator(self, accelerator: Accelerator) -> None:
        self._accelerator = accelerator

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = TorchCheckpointIO()
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def precision(self) -> Precision:
        return self._precision if self._precision is not None else Precision()

    @precision.setter
    def precision(self, precision: Optional[Precision]) -> None:
        self._precision = precision

    def _configure_launcher(self) -> None:
        """Attach the launcher based on Strategy."""

    def setup_environment(self) -> None:
        """Setup any processes or distributed connections.

        This must be called by the framework at the beginning of every process, before any distributed communication
        takes place.
        """
        assert self.accelerator is not None
        self.accelerator.setup_device(self.root_device)

    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Wraps the dataloader if necessary.

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`
        """
        return dataloader

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple[Module, List[Optimizer]]:
        """Set up a model and multiple optimizers together.

        The returned objects are expected to be in the same order they were passed in. The default implementation will
        call :meth:`setup_module` and :meth:`setup_optimizer` on the inputs.
        """
        module = self.setup_module(module)
        optimizers = [self.setup_optimizer(optimizer) for optimizer in optimizers]
        return module, optimizers

    def setup_module(self, module: Module) -> Module:
        """Performs setup for the model, e.g., by wrapping it by another class."""
        return module

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Performs setup for the optimizer, e.g., by wrapping it by another class."""
        return optimizer

    @abstractmethod
    def module_to_device(self, module: Module) -> None:
        """Moves the model to the correct device."""

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        """Moves the batch to the correct device.

        The returned batch is of the same type as the input batch, just
        having all tensors on the correct device.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
        """
        device = device or self.root_device
        return move_data_to_device(batch, device)

    def backward(self, tensor: Tensor, module: Optional[Module], *args: Any, **kwargs: Any) -> None:
        r"""Forwards backward-calls to the precision plugin."""
        self.precision.pre_backward(tensor, module)
        self.precision.backward(tensor, module, *args, **kwargs)
        self.precision.post_backward(tensor, module)

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        """Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            **kwargs: Any extra arguments to ``optimizer.step``
        """
        return self.precision.optimizer_step(optimizer, **kwargs)

    @abstractmethod
    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Perform an all_gather on all processes.

        Args:
            tensor: the tensor to all_gather
            group: the process group to gather results from
            sync_grads: flag that allows users to synchronize gradients for all_gather op
        """

    @abstractmethod
    def all_reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, Any]:
        """Reduces the given tensor (e.g. across GPUs/processes).

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to reduce
            reduce_op: the reduction operation. Defaults to 'mean'.
                Can also be a string 'sum' or ReduceOp.
        """

    @abstractmethod
    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronizes all processes which blocks processes until the whole group enters this function.

        Args:
            name: an optional name to pass into barrier.
        """

    @abstractmethod
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        """Broadcasts an object to all processes.

        Args:
            obj: the object to broadcast
            src: source rank
        """

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        """Reduce a boolean decision across all processes."""
        return decision

    def save_checkpoint(
        self, path: _PATH, state: Dict[str, Union[Module, Optimizer, Any]], storage_options: Optional[Any] = None
    ) -> None:
        """Save model, optimizer, and other state as a checkpoint file.

        Args:
            path: A path to where the file(s) should be saved
            state: A dictionary with contents to be saved. If the dict contains modules or optimizers, their
                state-dict will be retrieved and converted automatically.
            storage_options: Additional options for the ``CheckpointIO`` plugin
        """
        state = self._convert_stateful_objects_in_state(state)
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint=state, path=path, storage_options=storage_options)

    def get_module_state_dict(self, module: Module) -> Dict[str, Union[Any, Tensor]]:
        """Returns model state."""
        return module.state_dict()

    def get_optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        """Returns state of an optimizer.

        Allows for syncing/collating optimizer state from processes in custom plugins.
        """
        if hasattr(optimizer, "consolidate_state_dict"):
            # there are optimizers like PyTorch's ZeroRedundancyOptimizer that shard their
            # states, and to avoid OOM we consolidate the full state on rank 0 only
            optimizer.consolidate_state_dict()
            return optimizer.state_dict() if self.is_global_zero else {}

        # for optimizers that are not sharded, we return the state dict on all ranks
        return optimizer.state_dict()

    def load_checkpoint(
        self, path: _PATH, state: Optional[Dict[str, Union[Module, Optimizer, Any]]] = None
    ) -> Dict[str, Any]:
        """Load the contents from a checkpoint and restore the state of the given objects.

        Args:
            path: A path to where the file is located
            state: A dictionary of objects whose state will be restored in-place from the checkpoint path.
                If no state is given, then the checkpoint will be returned in full.

        Returns:
            The remaining items that were not restored into the given state dictionary. If no state dictionary is
            given, the full checkpoint will be returned.
        """
        torch.cuda.empty_cache()
        checkpoint = self.checkpoint_io.load_checkpoint(path)
        if not state:
            return checkpoint

        invalid_keys = [k for k in state if k not in checkpoint]
        if invalid_keys:
            # TODO(fabric): Make strict loading configurable to avoid this error if desired.
            raise KeyError(
                f"The requested state contains a key '{invalid_keys[0]}' that does not exist in the loaded checkpoint."
            )

        for name, obj in state.copy().items():
            if name not in checkpoint:
                continue
            elif isinstance(obj, _Stateful):
                if isinstance(obj, Module):
                    # TODO(fabric): Make strict loading configurable
                    obj.load_state_dict(checkpoint.pop(name), strict=True)
                else:
                    obj.load_state_dict(checkpoint.pop(name))
            else:
                state[name] = checkpoint.pop(name)
        return checkpoint

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint
        """
        if self.is_global_zero:
            self.checkpoint_io.remove_checkpoint(filepath)

    def teardown(self) -> None:
        """This method is called to teardown the training process.

        It is the right place to release memory and free other resources.
        """
        self.precision.teardown()
        assert self.accelerator is not None
        self.accelerator.teardown()
        self.checkpoint_io.teardown()

    def clip_gradients_norm(
        self,
        module: torch.nn.Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> torch.Tensor:
        """Clip gradients by norm."""
        self.precision.unscale_gradients(optimizer)
        parameters = self.precision.main_params(optimizer)
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite
        )

    def clip_gradients_value(self, module: torch.nn.Module, optimizer: Optimizer, clip_val: Union[float, int]) -> None:
        """Clip gradients by value."""
        self.precision.unscale_gradients(optimizer)
        parameters = self.precision.main_params(optimizer)
        return torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_val)

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        pass

    def _err_msg_joint_setup_required(self) -> str:
        return (
            f"The `{type(self).__name__}` does not support setting up the module and optimizer(s) independently."
            " Please call `setup_module_and_optimizers(model, [optimizer, ...])` to jointly set them up."
        )

    def _convert_stateful_objects_in_state(self, state: Dict[str, Union[Module, Optimizer, Any]]) -> Dict[str, Any]:
        converted_state = {}
        for key, obj in state.items():
            if isinstance(obj, Module):
                converted_state[key] = self.get_module_state_dict(module=obj)
            elif isinstance(obj, Optimizer):
                converted_state[key] = self.get_optimizer_state(optimizer=obj)
            else:
                converted_state[key] = obj
        return converted_state


class _BackwardSyncControl(ABC):
    """Interface for any :class:`Strategy` that wants to offer a functionality to enable or disable gradient
    synchronization during/after back-propagation.

    The most common use-case is gradient accumulation. If a :class:`Strategy` implements this interface, the user can
    implement their gradient accumulation loop very efficiently by disabling redundant gradient synchronization.
    """

    @contextmanager
    @abstractmethod
    def no_backward_sync(self, module: Module) -> Generator:
        """Blocks the synchronization of gradients during the backward pass.

        This is a context manager. It is only effective if it wraps a call to `.backward()`.
        """


class _Sharded(ABC):
    """Mixin-interface for any :class:`Strategy` that wants to expose functionality for sharding model
    parameters."""

    @abstractmethod
    @contextmanager
    def module_sharded_context(self) -> Generator:
        """A context manager that goes over the instantiation of an :class:`torch.nn.Module` and handles sharding
        of parameters on creation.

        By sharding layers directly on instantiation, one can reduce peak memory usage and initialization time.
        """
        yield

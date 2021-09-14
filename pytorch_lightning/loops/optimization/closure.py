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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.profiler import BaseProfiler, PassThroughProfiler
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.warnings import rank_zero_deprecation, WarningCache


@dataclass
class ClosureResult:
    """A container to hold the result of a :class:`AbstractClosure` call.

    It is created from the output of :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`.

    Attributes:
        closure_loss: The loss with a graph attached.
        loss: A detached copy of the closure loss.
        extra: Any keys other than the loss returned.
    """

    closure_loss: Optional[Tensor]
    loss: Optional[Tensor] = field(init=False, default=None)
    extra: Dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TODO: remove with the deprecation removal in v1.6
        ClosureResult._check_extra_detach_deprecation(self.extra)
        self.extra = recursive_detach(self.extra)

        self._clone_loss()

    def _clone_loss(self) -> None:
        if self.closure_loss is not None:
            # the loss will get scaled for amp. avoid any modifications to it
            self.loss = self.closure_loss.detach().clone()

    @classmethod
    def from_training_step_output(
        cls, training_step_output: Optional[STEP_OUTPUT], normalize: int = 1
    ) -> "ClosureResult":
        closure_loss, extra = None, {}

        if isinstance(training_step_output, dict):
            # this should not modify the `training_step_output`, as the user could be using it after `training_step_end`
            closure_loss = training_step_output.get("loss")
            extra = {k: v for k, v in training_step_output.items() if k not in ("loss", "hiddens")}
        elif isinstance(training_step_output, Tensor):
            closure_loss = training_step_output

        if closure_loss is not None:
            # accumulate the loss. If ``accumulate_grad_batches == 1``, no effect
            closure_loss /= normalize

        return cls(closure_loss, extra=extra)

    @staticmethod
    def _check_extra_detach_deprecation(extra: Dict[str, Any]) -> None:
        def check_fn(v: Tensor) -> Tensor:
            if v.grad_fn is not None:
                rank_zero_deprecation(
                    f"One of the returned values {set(extra.keys())} has a `grad_fn`. We will detach it automatically"
                    " but this behaviour will change in v1.6. Please detach it manually:"
                    " `return {'loss': ..., 'something': something.detach()}`"
                )
            return v

        apply_to_collection(extra, Tensor, check_fn)

    def drop_closure_loss(self) -> "ClosureResult":
        """Return itself without the closure loss which could have a `grad_fn`"""
        self.closure_loss = None
        return self


class AbstractClosure(ABC):
    """Abstract base class for optimizer closures in Lightning.

    Formally, a closure is binding variables from an external scope to a function that does a computation on these
    variables without taking them explicitly as input. This has the benefit that a closure can be passed to an
    object which later can call it like a function but without requiring to pass in any arguments.

    This class provides a simple abstraction making the instance of this class callable like a function while capturing
    the :class:`ClosureResult` and caching it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._result: Optional[ClosureResult] = None

    def consume_result(self) -> ClosureResult:
        """The cached result from the last time the closure was called.

        Once accessed, the internal reference gets reset and the consumer will have to hold on to the reference as long
        as necessary.
        """
        if self._result is None:
            raise MisconfigurationException(
                "The closure hasn't been executed."
                " HINT: did you call `optimizer_closure()` in your `optimizer_step` hook? It could also happen because"
                " the `optimizer.step(optimizer_closure)` call did not execute it internally."
            )
        result, self._result = self._result, None  # free memory
        return result

    @abstractmethod
    def closure(self, *args: Any, **kwargs: Any) -> ClosureResult:
        """Implements the behavior of the closure once it is getting called."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Tensor]:
        self._result = self.closure(*args, **kwargs)
        return self._result.loss


class Closure(AbstractClosure):
    """An implementation of a :class:`AbstractClosure` for optimization in Lightning that combines three elementary
    closures into one: ``training_step``, ``backward`` and ``zero_grad``.

    The Closure gets created by the training loop(s) and is then passed to the
    :meth:`torch.optim.Optimizer.step` method. An optimizer is responsible for calling the closure and optionally
    do something with the output.

    Args:
        step_fn: This is typically the :meth:`pytorch_lightning.core.lightning.LightningModule.training_step
            wrapped with processing for its outputs
        backward_fn: A function that takes a loss value as input, performs back-propagation and returns the loss value.
            Can be set to ``None`` to skip the backward operation.
        zero_grad_fn: A function that zeroes the gradients. Can be set to ``None`` to skip zero_grad, for example
            when accumulating gradients.
        profiler: A profiler for profiling the actions of the passed in closure functions.

    Example:

        closure = Closure()
        optimizer = torch.optim.Adam(...)
        optimizer.step(closure)
    """

    warning_cache = WarningCache()

    def __init__(
        self,
        step_fn: Callable[[], ClosureResult],
        backward_fn: Optional[Callable[[Tensor], Tensor]] = None,
        zero_grad_fn: Optional[Callable[[], None]] = None,
        profiler: Optional[BaseProfiler] = None,
    ):
        super().__init__()
        self._step_fn = step_fn
        self._backward_fn = backward_fn
        self._zero_grad_fn = zero_grad_fn
        self._profiler = PassThroughProfiler() if profiler is None else profiler

    def closure(self, *args: Any, **kwargs: Any) -> ClosureResult:
        with self._profiler.profile("training_step_and_backward"):
            step_output = self._step_fn()

            if step_output.closure_loss is None:
                self.warning_cache.warn(
                    "`training_step` returned `None`. If this was on purpose, ignore this warning..."
                )

            if self._zero_grad_fn is not None:
                with self._profiler.profile("zero_grad"):
                    self._zero_grad_fn()

            if self._backward_fn is not None and step_output.closure_loss is not None:
                with self._profiler.profile("backward"):
                    step_output.closure_loss = self._backward_fn(step_output.closure_loss)

        return step_output


class _ClosureExecutor:

    """This class is used to prevent fault tolerant to create a checkpoint while parameters are being updated."""

    def __init__(self, closure: Closure, trainer: "pl.Trainer"):
        self._closure = closure
        self._trainer = trainer
        self._fault_tolerant_possible: Optional[bool] = None

    def consume_result(self) -> ClosureResult:
        return self._closure.consume_result()

    def __call__(self):
        # enable fault tolerant during closure execution
        with self._trainer._fault_tolerant_supported(enable=True):
            loss = self._closure()
        self._fault_tolerant_possible = self._trainer._fault_tolerant_possible
        # prevent fault tolerant during parameters update.
        self._trainer._fault_tolerant_possible = False
        return loss

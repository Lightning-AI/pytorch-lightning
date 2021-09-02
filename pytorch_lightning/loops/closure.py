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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from torch import Tensor

from pytorch_lightning.profiler import BaseProfiler, PassThroughProfiler
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.utilities.warnings import WarningCache


@dataclass
class ClosureResult:
    """A container to hold the result of a :class:`AbstractClosure` call.

    Attributes:
        closure_loss: The loss with a graph attached.
        loss: A detached copy of the closure loss.
        result_collection: A collection of results returned by the closure.
    """

    closure_loss: Optional[Tensor]
    loss: Optional[Tensor]
    result_collection: Optional[ResultCollection]


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

    def get_result(self) -> Optional[ClosureResult]:
        """The cached result from the last time the closure was called.

        Once accessed, the internal reference gets reset and the consumer will have to hold on to the reference as long
        as necessary.
        """
        result = self._result
        self._result = None  # free memory
        return result

    @abstractmethod
    def closure(self, *args: Any, **kwargs: Any) -> Optional[ClosureResult]:
        """Implements the behavior of the closure once it is getting called."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Tensor]:
        self._result = self.closure(*args, **kwargs)
        if self._result is not None:
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
        step_fn: Callable[[], Optional[Dict]],
        backward_fn: Optional[Callable[[Tensor], Tensor]] = None,
        zero_grad_fn: Optional[Callable[[], None]] = None,
        profiler: Optional[BaseProfiler] = None,
    ):
        super().__init__()
        self._step_fn = step_fn
        self._backward_fn = backward_fn
        self._zero_grad_fn = zero_grad_fn
        self._profiler = PassThroughProfiler() if profiler is None else profiler

    def closure(self, *args: Any, **kwargs: Any) -> Optional[ClosureResult]:
        with self._profiler.profile("training_step_and_backward"):
            step_output = self._step_fn()
            step_output = ClosureResult(**step_output) if step_output else None

            if step_output is None:
                self.warning_cache.warn("training_step returned None. If this was on purpose, ignore this warning...")

            if self._zero_grad_fn is not None:
                with self._profiler.profile("zero_grad"):
                    self._zero_grad_fn()

            if self._backward_fn is not None and step_output is not None and step_output.closure_loss is not None:
                with self._profiler.profile("backward"):
                    step_output.closure_loss = self._backward_fn(step_output.closure_loss)

        return step_output

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
from typing import Any, Dict, Optional

from torch import Tensor

from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warnings import rank_zero_deprecation


@dataclass
class OutputResult:
    @staticmethod
    def _check_extra_detach_deprecation(extra: Dict[str, Any]) -> None:
        # TODO: remove with the deprecation removal in v1.6
        # this is only here to avoid duplication
        def check_fn(v: Tensor) -> Tensor:
            if v.grad_fn is not None:
                rank_zero_deprecation(
                    f"One of the returned values {set(extra.keys())} has a `grad_fn`. We will detach it automatically"
                    " but this behaviour will change in v1.6. Please detach it manually:"
                    " `return {'loss': ..., 'something': something.detach()}`"
                )
            return v

        apply_to_collection(extra, Tensor, check_fn)


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
        self._result: Optional[OutputResult] = None

    def consume_result(self) -> OutputResult:
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
    def closure(self, *args: Any, **kwargs: Any) -> OutputResult:
        """Implements the behavior of the closure once it is getting called."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self._result = self.closure(*args, **kwargs)

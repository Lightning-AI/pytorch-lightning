from abc import abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, Callable, Tuple, Any, Dict

from torch import Tensor
from torch.optim import Optimizer

from pytorch_lightning.profiler import BaseProfiler
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache


@dataclass
class ClosureResult:
    output: Optional[STEP_OUTPUT] = None

    @property
    def loss(self) -> Optional[Tensor]:
        if isinstance(self.output, dict):
            return self.output.get("loss")
        elif isinstance(self.output, Tensor):
            return self.output


class Closure:

    def __init__(self):
        super().__init__()
        self._result = ClosureResult()

    @property
    def result(self) -> ClosureResult:
        """ The cached result from the last time the closure was called. """
        return self._result

    @abstractmethod
    def closure(self, *args, **kwargs):
        """ Implements the behavior of the closure once it is getting called. """
        pass

    def __call__(self, *args, **kwargs) -> Optional[Tensor]:
        output = self.closure(*args, **kwargs)
        self._result = ClosureResult(output)
        return self._result.loss


class LightningClosure(Closure):
    """ Training step and backward closure. """

    warning_cache: Optional[WarningCache] = None

    def __init__(
        self,
        step_fn: Callable,
        backward_fn: Callable,  # set to None for manual or skip_backward
        zero_grad_fn: Callable,  # set to None if accumulating
        profiler: BaseProfiler,
    ):
        super().__init__()
        self._step_fn = step_fn
        self._backward_fn = backward_fn
        self._zero_grad_fn = zero_grad_fn
        self._profiler = profiler
        if self.warning_cache is None:
            self.warning_cache = WarningCache()

    def closure(self, *args, **kwargs) -> STEP_OUTPUT:
        with self._profiler.profile("training_step_and_backward"):
            # lightning module hook
            output = self._step_fn()

            if output is None:
                self.warning_cache.warn(
                    "training_step returned None. If this was on purpose, ignore this warning..."
                )

            if self._zero_grad_fn is not None:
                with self._profiler.profile("zero_grad"):
                    self._zero_grad_fn()

            if self._backward_fn is not None and output is not None:
                with self._profiler.profile("backward"):
                    self._backward_fn(output)

        return output

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable

from torch import Tensor

from pytorch_lightning.profiler import BaseProfiler
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.utilities.warnings import WarningCache


@dataclass
class ClosureResult:
    closure_loss: Tensor
    result_collection: ResultCollection

    @property
    def loss(self):
        if self.closure_loss is not None:
            return self.closure_loss.detach().clone()


class Closure:
    def __init__(self):
        super().__init__()
        self._result = None

    @property
    def result(self) -> ClosureResult:
        """The cached result from the last time the closure was called."""
        return self._result

    @abstractmethod
    def closure(self, *args, **kwargs):
        """Implements the behavior of the closure once it is getting called."""
        pass

    def __call__(self, *args, **kwargs) -> Optional[Tensor]:
        self._result = self.closure(*args, **kwargs)
        if self._result is not None:
            return self._result.loss


class LightningClosure(Closure):
    """Training step and backward closure."""

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

    def closure(self, *args, **kwargs) -> ClosureResult:
        with self._profiler.profile("training_step_and_backward"):
            output = self._step_fn()
            output = ClosureResult(**output) if output else None

            if output is None:
                self.warning_cache.warn("training_step returned None. If this was on purpose, ignore this warning...")

            if self._zero_grad_fn is not None:
                with self._profiler.profile("zero_grad"):
                    self._zero_grad_fn()

            if self._backward_fn is not None and output is not None:
                with self._profiler.profile("backward"):
                    self._backward_fn(output)

        return output

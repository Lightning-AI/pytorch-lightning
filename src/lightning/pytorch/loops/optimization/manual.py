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
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor
from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import do_nothing_closure
from lightning.pytorch.loops import _Loop
from lightning.pytorch.loops.optimization.closure import OutputResult
from lightning.pytorch.loops.progress import _Progress, _ReadyCompletedTracker
from lightning.pytorch.trainer import call
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT


@dataclass
class ManualResult(OutputResult):
    """A container to hold the result returned by ``_ManualOptimization``.

    It is created from the output of :meth:`~lightning.pytorch.core.LightningModule.training_step`.

    Attributes:
        extra: Anything returned by the ``training_step``.

    """

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_training_step_output(cls, training_step_output: STEP_OUTPUT) -> "ManualResult":
        extra = {}
        if isinstance(training_step_output, dict):
            extra = training_step_output.copy()
        elif isinstance(training_step_output, Tensor):
            extra = {"loss": training_step_output}
        elif training_step_output is not None:
            raise MisconfigurationException(
                "In manual optimization, `training_step` must either return a Tensor or have no return."
            )

        if "loss" in extra:
            # we detach manually as it's expected that it will have a `grad_fn`
            extra["loss"] = extra["loss"].detach()

        return cls(extra=extra)

    @override
    def asdict(self) -> dict[str, Any]:
        return self.extra


_OUTPUTS_TYPE = dict[str, Any]


class _ManualOptimization(_Loop):
    """A special loop implementing what is known in Lightning as Manual Optimization where the optimization happens
    entirely in the :meth:`~lightning.pytorch.core.LightningModule.training_step` and therefore the user is responsible
    for back-propagating gradients and making calls to the optimizers.

    This loop is a trivial case because it performs only a single iteration (calling directly into the module's
    :meth:`~lightning.pytorch.core.LightningModule.training_step`) and passing through the output(s).

    """

    output_result_cls = ManualResult

    def __init__(self, trainer: "pl.Trainer") -> None:
        super().__init__(trainer)
        # since manual optimization does not track lr scheduler or optimizer frequencies, we use a simpler progress than
        # `_OptimizationProgress`
        self.optim_step_progress = _Progress.from_defaults(_ReadyCompletedTracker)

        self._output: _OUTPUTS_TYPE = {}

    def run(self, kwargs: OrderedDict) -> _OUTPUTS_TYPE:
        self.on_run_start()
        with suppress(StopIteration):  # no loop to break at this level
            self.advance(kwargs)
        self._restarting = False
        return self.on_run_end()

    def on_run_start(self) -> None:
        # inject logic around the optimizer step
        for lightning_optimizer in self.trainer.strategy._lightning_optimizers:
            lightning_optimizer._on_before_step = self._on_before_step
            lightning_optimizer._on_after_step = self._on_after_step

    def advance(self, kwargs: OrderedDict) -> None:
        """Performs the training step for manual optimization.

        Args:
            kwargs: The kwargs passed down to the hooks.

        """
        trainer = self.trainer

        # manually capture logged metrics
        training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
        del kwargs  # release the batch from memory
        self.trainer.strategy.post_training_step()  # unused hook - call anyway for backward compatibility
        result = self.output_result_cls.from_training_step_output(training_step_output)

        self._output = result.asdict()

    def on_run_end(self) -> _OUTPUTS_TYPE:
        """Returns the result of this loop, i.e., the post-processed outputs from the training step."""
        output, self._output = self._output, {}  # free memory
        # reset logic around the optimizer step
        for lightning_optimizer in self.trainer.strategy._lightning_optimizers:
            lightning_optimizer._on_before_step = do_nothing_closure
            lightning_optimizer._on_after_step = do_nothing_closure
        return output

    def _on_before_step(self) -> None:
        self.optim_step_progress.increment_ready()
        self.trainer.profiler.start("optimizer_step")

    def _on_after_step(self) -> None:
        self.trainer.profiler.stop("optimizer_step")
        self.optim_step_progress.increment_completed()

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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor

from pytorch_lightning.core.optimizer import do_nothing_closure
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.optimization.closure import OutputResult
from pytorch_lightning.loops.utilities import _build_training_step_kwargs, _extract_hiddens
from pytorch_lightning.trainer.progress import Progress, ReadyCompletedTracker
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT


@dataclass
class ManualResult(OutputResult):
    """A container to hold the result returned by the ``ManualLoop``.

    It is created from the output of :meth:`~pytorch_lightning.core.module.LightningModule.training_step`.

    Attributes:
        extra: Anything returned by the ``training_step``.
    """

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_training_step_output(cls, training_step_output: Optional[STEP_OUTPUT]) -> "ManualResult":
        extra = {}
        if isinstance(training_step_output, dict):
            extra = {k: v for k, v in training_step_output.items() if k != "hiddens"}
        elif isinstance(training_step_output, Tensor):
            extra = {"loss": training_step_output}
        elif training_step_output is not None:
            raise MisconfigurationException(
                "In manual optimization, `training_step` must either return a Tensor, "
                "a dict with extras to pass to `training_epoch_end` or have no return."
            )

        if "loss" in extra:
            # we detach manually as it's expected that it will have a `grad_fn`
            extra["loss"] = extra["loss"].detach()

        return cls(extra=extra)

    def asdict(self) -> Dict[str, Any]:
        return self.extra


_OUTPUTS_TYPE = Dict[str, Any]


class ManualOptimization(Loop[_OUTPUTS_TYPE]):
    """A special loop implementing what is known in Lightning as Manual Optimization where the optimization happens
    entirely in the :meth:`~pytorch_lightning.core.module.LightningModule.training_step` and therefore the user is
    responsible for back-propagating gradients and making calls to the optimizers.

    This loop is a trivial case because it performs only a single iteration (calling directly into the module's
    :meth:`~pytorch_lightning.core.module.LightningModule.training_step`) and passing through the output(s).
    """

    output_result_cls = ManualResult

    def __init__(self) -> None:
        super().__init__()
        # since manual optimization does not track lr scheduler or optimizer frequencies, we use a simpler progress than
        # `OptimizationProgress`
        self.optim_step_progress = Progress.from_defaults(ReadyCompletedTracker)

        self._done: bool = False
        self._hiddens: Optional[Any] = None
        self._output: _OUTPUTS_TYPE = {}

    @property
    def done(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._done = False

    def on_run_start(self, *_: Any, **__: Any) -> None:
        # inject logic around the optimizer step
        for i, lightning_optimizer in self.trainer.strategy._lightning_optimizers.items():
            lightning_optimizer._on_before_step = self._on_before_step
            lightning_optimizer._on_after_step = self._on_after_step

    def advance(self, kwargs: OrderedDict) -> None:  # type: ignore[override]
        """Performs the training step for manual optimization.

        Args:
            kwargs: The kwargs passed down to the hooks.
        """
        kwargs = self._build_kwargs(kwargs, self._hiddens)

        # manually capture logged metrics
        training_step_output = self.trainer._call_strategy_hook("training_step", *kwargs.values())
        del kwargs  # release the batch from memory
        self.trainer.strategy.post_training_step()

        model_output = self.trainer._call_lightning_module_hook("training_step_end", training_step_output)
        strategy_output = self.trainer._call_strategy_hook("training_step_end", training_step_output)
        training_step_output = strategy_output if model_output is None else model_output
        self._hiddens = _extract_hiddens(training_step_output, self.trainer.lightning_module.truncated_bptt_steps)

        result = self.output_result_cls.from_training_step_output(training_step_output)

        if self.trainer.move_metrics_to_cpu:
            # hiddens and the training step output are not moved as they are not considered "metrics"
            # the user might need them on the correct device for an operation in `training_epoch_end`
            assert self.trainer._results is not None
            self.trainer._results.cpu()

        self._done = True
        self._output = result.asdict()

    def on_run_end(self) -> _OUTPUTS_TYPE:
        """Returns the result of this loop, i.e., the post-processed outputs from the training step."""
        output, self._output = self._output, {}  # free memory
        # reset logic around the optimizer step
        for i, lightning_optimizer in self.trainer.strategy._lightning_optimizers.items():
            lightning_optimizer._on_before_step = do_nothing_closure
            lightning_optimizer._on_after_step = do_nothing_closure
        return output

    def _on_before_step(self) -> None:
        self.optim_step_progress.increment_ready()
        self.trainer.profiler.start("optimizer_step")

    def _on_after_step(self) -> None:
        self.trainer.profiler.stop("optimizer_step")
        self.optim_step_progress.increment_completed()

    def _build_kwargs(self, kwargs: OrderedDict, hiddens: Optional[Any]) -> OrderedDict:
        """Helper method to build the arguments for the current step.

        Args:
            kwargs: The kwargs passed down to the hooks.
            hiddens: the hidden state of the previous RNN iteration.

        Returns:
            The kwargs passed down to the hooks.
        """
        return _build_training_step_kwargs(
            kwargs, self.trainer.lightning_module, self.trainer.optimizers, None, hiddens
        )

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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor

from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.optimization.closure import OutputResult
from pytorch_lightning.loops.utilities import _build_training_step_kwargs, _extract_hiddens, check_finite_loss
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.types import STEP_OUTPUT


@dataclass
class ManualResult(OutputResult):
    """A container to hold the result returned by the ``ManualLoop``.

    It is created from the output of :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`.

    Attributes:
        extra: Anything returned by the ``training_step``.
    """

    extra: Dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TODO: remove with the deprecation removal in v1.6
        self._check_extra_detach_deprecation(self.extra)
        self.extra = recursive_detach(self.extra)

    @classmethod
    def from_training_step_output(
        cls, training_step_output: Optional[STEP_OUTPUT], normalize: int = 1
    ) -> "ManualResult":
        extra = {}
        if isinstance(training_step_output, dict):
            extra = {k: v for k, v in training_step_output.items() if k != "hiddens"}
        elif isinstance(training_step_output, Tensor):
            extra["loss"] = training_step_output

        if "loss" in extra:
            # accumulate the loss. If ``accumulate_grad_batches == 1``, no effect
            extra["loss"] = extra["loss"].detach().div(normalize)

        return cls(extra=extra)


class ManualOptimization(Loop):
    """A special loop implementing what is known in Lightning as Manual Optimization where the optimization happens
    entirely in the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` and therefore the user
    is responsible for back-propagating gradients and making calls to the optimizers.

    This loop is a trivial case because it performs only a single iteration (calling directly into the module's
    :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`) and passing through the output(s).
    """

    def __init__(self) -> None:
        super().__init__()
        self._done: bool = False
        self._hiddens: Optional[Any] = None
        self._output: Optional[ManualResult] = None

    @property
    def done(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._done = False

    def advance(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        """Performs the training step for manual optimization.

        Args:
            batch: the current tbptt split of the current batch
            batch_idx: the index of the current batch
        """
        assert self.trainer is not None
        lightning_module = self.trainer.lightning_module

        with self.trainer.profiler.profile("model_forward"):

            step_kwargs = _build_training_step_kwargs(
                lightning_module, self.trainer.optimizers, batch, batch_idx, opt_idx=None, hiddens=self._hiddens
            )

            # manually capture logged metrics
            lightning_module._current_fx_name = "training_step"
            with self.trainer.profiler.profile("training_step"):
                training_step_output = self.trainer.accelerator.training_step(step_kwargs)
                self.trainer.accelerator.post_training_step()

            del step_kwargs

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            self._hiddens = _extract_hiddens(training_step_output, lightning_module.truncated_bptt_steps)

            result = ManualResult.from_training_step_output(training_step_output, self.trainer.accumulate_grad_batches)

            if self.trainer.terminate_on_nan:
                check_finite_loss(result.closure_loss)

            if self.trainer.move_metrics_to_cpu:
                # hiddens and the training step output are not moved as they are not considered "metrics"
                # the user might need them on the correct device for an operation in `training_epoch_end`
                assert self.trainer._results is not None
                self.trainer._results.cpu()

        self._done = True
        self._output = result

    def on_run_end(self) -> Optional[ManualResult]:
        """Returns the result of this loop, i.e., the post-processed outputs from the training step."""
        output, self._output = self._output, None  # free memory
        # #9052 added support for raising `StopIteration` in the `training_step`. If that happens, then `advance`
        # doesn't finish and `self._output` stays as `None`. If #9415 happens then this would always return a result
        return output

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
from pytorch_lightning.loops.utilities import _build_training_step_kwargs, _extract_hiddens
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT


@dataclass
class ManualResult(OutputResult):
    """A container to hold the result returned by the ``ManualLoop``.

    It is created from the output of :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`.

    Attributes:
        extra: Anything returned by the ``training_step``.
    """

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TODO: remove with the deprecation removal in v1.6
        self.extra = self._check_extra_detach_deprecation(self.extra)

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
    entirely in the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` and therefore the user
    is responsible for back-propagating gradients and making calls to the optimizers.

    This loop is a trivial case because it performs only a single iteration (calling directly into the module's
    :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`) and passing through the output(s).
    """

    output_result_cls = ManualResult

    def __init__(self) -> None:
        super().__init__()
        self._done: bool = False
        self._hiddens: Optional[Any] = None
        self._output: _OUTPUTS_TYPE = {}

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
                self.trainer.training_type_plugin.post_training_step()

            del step_kwargs

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            self._hiddens = _extract_hiddens(training_step_output, lightning_module.truncated_bptt_steps)

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
        return output

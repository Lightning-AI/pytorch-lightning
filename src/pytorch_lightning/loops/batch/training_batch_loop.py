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
from typing import Any, List, Optional, Tuple, Union

from torch import Tensor
from typing_extensions import OrderedDict

from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.loops.optimization.manual_loop import _OUTPUTS_TYPE as _MANUAL_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.optimization.manual_loop import ManualOptimization
from pytorch_lightning.loops.optimization.optimizer_loop import _OUTPUTS_TYPE as _OPTIMIZER_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.optimization.optimizer_loop import OptimizerLoop
from pytorch_lightning.loops.utilities import _get_active_optimizers
from pytorch_lightning.trainer.supporters import TensorRunningAccum

_OUTPUTS_TYPE = List[Union[_OPTIMIZER_LOOP_OUTPUTS_TYPE, _MANUAL_LOOP_OUTPUTS_TYPE]]


class TrainingBatchLoop(Loop[_OUTPUTS_TYPE]):
    """Runs over a single batch of data."""

    def __init__(self) -> None:
        super().__init__()
        self.accumulated_loss = TensorRunningAccum(window_length=20)
        self.running_loss = TensorRunningAccum(window_length=20)
        # the current split index when the batch gets split into chunks in truncated backprop through time
        self.split_idx: int = 0
        self.optimizer_loop = OptimizerLoop()
        self.manual_loop = ManualOptimization()

        self._outputs: _OUTPUTS_TYPE = []
        self._remaining_splits: List[Tuple[int, Any]] = []

    @property
    def done(self) -> bool:
        """Returns if all batch splits have been processed already."""
        return len(self._remaining_splits) == 0

    def connect(  # type: ignore[override]
        self, optimizer_loop: Optional[OptimizerLoop] = None, manual_loop: Optional[ManualOptimization] = None
    ) -> None:
        if optimizer_loop is not None:
            self.optimizer_loop = optimizer_loop
        if manual_loop is not None:
            self.manual_loop = manual_loop

    def reset(self) -> None:
        """Resets the loop state."""
        self._outputs = []

    def on_run_start(self, kwargs: OrderedDict) -> None:
        """Splits the data into tbptt splits.

        Args:
            kwargs: the kwargs passed down to the hooks.
        """
        batch = kwargs["batch"]
        self._remaining_splits = list(enumerate(self._tbptt_split_batch(batch)))

    def advance(self, kwargs: OrderedDict) -> None:
        """Runs the train step together with optimization (if necessary) on the current batch split.

        Args:
            kwargs: the kwargs passed down to the hooks.
        """
        # replace the batch with the split batch
        self.split_idx, kwargs["batch"] = self._remaining_splits.pop(0)

        self.trainer._logger_connector.on_train_split_start(self.split_idx)

        outputs: Optional[Union[_OPTIMIZER_LOOP_OUTPUTS_TYPE, _MANUAL_LOOP_OUTPUTS_TYPE]] = None  # for mypy
        # choose which loop will run the optimization
        if self.trainer.lightning_module.automatic_optimization:
            optimizers = _get_active_optimizers(
                self.trainer.optimizers, self.trainer.optimizer_frequencies, kwargs.get("batch_idx", 0)
            )
            outputs = self.optimizer_loop.run(optimizers, kwargs)
        else:
            outputs = self.manual_loop.run(kwargs)
        if outputs:
            # automatic: can be empty if all optimizers skip their batches
            # manual: #9052 added support for raising `StopIteration` in the `training_step`. If that happens,
            # then `advance` doesn't finish and an empty dict is returned
            self._outputs.append(outputs)

    def on_run_end(self) -> _OUTPUTS_TYPE:
        self.optimizer_loop._hiddens = None
        # this is not necessary as the manual loop runs for only 1 iteration, but just in case
        self.manual_loop._hiddens = None
        output, self._outputs = self._outputs, []  # free memory
        self._remaining_splits = []
        return output

    def teardown(self) -> None:
        self.optimizer_loop.teardown()
        self.manual_loop.teardown()
        # release memory
        if self.accumulated_loss.memory is not None:
            self.accumulated_loss.memory = self.accumulated_loss.memory.cpu()
        if self.running_loss.memory is not None:
            self.running_loss.memory = self.running_loss.memory.cpu()

    def _tbptt_split_batch(self, batch: Any) -> List[Any]:
        """Splits a single batch into a list of sequence steps for tbptt.

        Args:
            batch: the current batch to split
        """
        tbptt_steps = self.trainer.lightning_module.truncated_bptt_steps
        if tbptt_steps == 0:
            return [batch]

        splits = self.trainer._call_lightning_module_hook("tbptt_split_batch", batch, tbptt_steps)
        return splits

    def _update_running_loss(self, current_loss: Tensor) -> None:
        """Updates the running loss value with the current value."""
        if self.trainer.lightning_module.automatic_optimization:
            # track total loss for logging (avoid mem leaks)
            self.accumulated_loss.append(current_loss)

        accumulated_loss = self.accumulated_loss.mean()

        if accumulated_loss is not None:
            # calculate running loss for display
            self.running_loss.append(self.accumulated_loss.mean() * self.trainer.accumulate_grad_batches)

        # reset for next set of accumulated grads
        self.accumulated_loss.reset()

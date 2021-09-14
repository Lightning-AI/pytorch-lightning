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
from typing import Any, List, Optional, Tuple

import numpy as np
from deprecate import void
from torch import Tensor
from torch.optim import Optimizer

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.optimization.manual_loop import ManualOptimization
from pytorch_lightning.loops.optimization.optimizer_loop import OptimizerLoop
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache


class TrainingBatchLoop(Loop):
    """Runs over a single batch of data."""

    def __init__(self) -> None:
        super().__init__()
        self.accumulated_loss: Optional[Tensor] = None
        self.batch_outputs: Optional[List[List[STEP_OUTPUT]]] = None
        self.running_loss: TensorRunningAccum = TensorRunningAccum(window_length=20)
        # the current split index when the batch gets split into chunks in truncated backprop through time
        self.split_idx: Optional[int] = None
        self.optimizer_loop = OptimizerLoop()
        self.manual_loop = ManualOptimization()

        self._warning_cache: WarningCache = WarningCache()
        self._optimizer_freq_cumsum: Optional[int] = None
        self._remaining_splits: Optional[List[Any]] = None

    @property
    def done(self) -> bool:
        """Returns if all batch splits have been processed already."""
        return len(self._remaining_splits) == 0

    @property
    def optimizer_freq_cumsum(self) -> int:
        """Returns the cumulated sum of optimizer frequencies."""
        if self._optimizer_freq_cumsum is None:
            self._optimizer_freq_cumsum = np.cumsum(self.trainer.optimizer_frequencies)
        return self._optimizer_freq_cumsum

    def connect(
        self, optimizer_loop: Optional["Loop"] = None, manual_loop: Optional[ManualOptimization] = None
    ) -> None:
        if optimizer_loop is not None:
            self.optimizer_loop = optimizer_loop
        if manual_loop is not None:
            self.manual_loop = manual_loop

    def run(self, batch: Any, batch_idx: int) -> AttributeDict:
        """Runs all the data splits and the ``on_batch_start`` and ``on_train_batch_start`` hooks.

        Args:
            batch: the current batch to run the train step on
            batch_idx: the index of the current batch
        """
        if batch is None:
            self._warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
            return AttributeDict(signal=0, training_step_output=[[]])

        # hook
        self.trainer.logger_connector.on_batch_start()
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1)

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, 0)
        if response == -1:
            return AttributeDict(signal=-1)

        self.trainer.fit_loop.epoch_loop.batch_progress.increment_started()

        super().run(batch, batch_idx)
        output = AttributeDict(signal=0, training_step_output=self.batch_outputs)
        self.batch_outputs = None  # free memory
        return output

    def reset(self) -> None:
        """Resets the loop state."""
        self.batch_outputs = [[] for _ in range(len(self.trainer.optimizers))]

    def on_run_start(self, batch: Any, batch_idx: int):
        """Splits the data into tbptt splits.

        Args:
            batch: the current batch to run the trainstep on
            batch_idx: the index of the current batch
        """
        void(batch_idx)
        self._remaining_splits = list(enumerate(self._tbptt_split_batch(batch)))

    def advance(self, batch, batch_idx):
        """Runs the train step together with optimization (if necessary) on the current batch split.

        Args:
            batch: the current batch to run the training on (this is not the split!)
            batch_idx: the index of the current batch
        """
        void(batch)
        split_idx, split_batch = self._remaining_splits.pop(0)
        self.split_idx = split_idx

        # let logger connector extract current batch size
        self.trainer.logger_connector.on_train_split_start(batch_idx, split_idx, split_batch)

        if self.trainer.lightning_module.automatic_optimization:
            # in automatic optimization, hand over execution to the OptimizerLoop
            optimizers = [optimizer for _, optimizer in self.get_active_optimizers(batch_idx)]
            batch_outputs = self.optimizer_loop.run(split_batch, optimizers, batch_idx)
            # combine outputs from each optimizer
            for k in range(len(batch_outputs)):
                self.batch_outputs[k].extend(batch_outputs[k])
        else:
            # in manual optimization, hand over execution to the ManualOptimization loop
            result = self.manual_loop.run(split_batch, batch_idx)
            if result:
                self.batch_outputs[0].append(result)

    def on_run_end(self) -> None:
        self.optimizer_loop._hiddens = None
        # this is not necessary as the manual loop runs for only 1 iteration, but just in case
        self.manual_loop._hiddens = None

    def teardown(self) -> None:
        # release memory
        self._remaining_splits = None

    def num_active_optimizers(self, batch_idx: Optional[int] = None) -> int:
        """Gets the number of active optimizers based on their frequency."""
        return len(self.get_active_optimizers(batch_idx))

    def _tbptt_split_batch(self, batch: Any) -> List[Any]:
        """Splits a single batch into a list of sequence steps for tbptt.

        Args:
            batch: the current batch to split
        """
        tbptt_steps = self.trainer.lightning_module.truncated_bptt_steps
        if tbptt_steps == 0:
            return [batch]

        model_ref = self.trainer.lightning_module
        with self.trainer.profiler.profile("tbptt_split_batch"):
            splits = model_ref.tbptt_split_batch(batch, tbptt_steps)
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

    def get_active_optimizers(self, batch_idx: Optional[int] = None) -> List[Tuple[int, Optimizer]]:
        """Returns the currently active optimizers. When multiple optimizers are used with different frequencies,
        only one of the optimizers is active at a time.

        Returns:
            A list of tuples (opt_idx, optimizer) of currently active optimizers.
        """
        if not self.trainer.optimizer_frequencies:
            # call training_step once per optimizer
            return list(enumerate(self.trainer.optimizers))

        optimizers_loop_length = self.optimizer_freq_cumsum[-1]
        current_place_in_loop = batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = int(np.argmax(self.optimizer_freq_cumsum > current_place_in_loop))
        return [(opt_idx, self.trainer.optimizers[opt_idx])]

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
import logging
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

import pytorch_lightning as pl
from pytorch_lightning.loops.utilities import (
    _check_training_step_output,
    _process_training_step_output,
    check_finite_loss,
)
from pytorch_lightning.trainer.progress import OptimizationProgress
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature

log = logging.getLogger(__name__)


class IteratorBatchProcessor:
    """
    The processor for performing a training iteration when ``training_step`` needs access to the
    dataloader. It is selected when the signature of ``training_step`` contains ``dataloader_iter``:

        def training_step(self, dataloader_iter: Iterator) -> STEP_OUTPUT:

    The ``training_step`` is allowed to fetch multiple batches during one training iteration. The
    framework provides minimum amount of automation with regards to model optimization. The
    flexibility allows for ease of experimentation with inter-batch parallelism techniques.

    This processor doesn't support ``automatic_optimization`` and ``tbptt``. An error will be thrown
    if the ``LightningModule`` or the ``Trainer`` is configured to use these features.

    The ``training_step`` is responsible for reporting whether it has reached the last batch by
    including an ``is_last`` field in the result dict. Failing to do so will result in an error.

    The ``training_step`` should only optimize the model with one batch for the sake of API and
    reporting consistency (TODO: consider removing this limitation).

    Args:
        trainer: a reference to the trainer
        model: a reference to the lightning module (for config validation purposes only)
    """

    def __init__(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        if is_overridden("on_train_batch_start", model):
            raise MisconfigurationException(
                "The model hook `on_train_batch_start` is not compatible with "
                "taking a `dataloader_iter` argument in your `training_step`."
            )
        if is_overridden("on_train_batch_end", model):
            raise MisconfigurationException(
                "The model hook `on_train_batch_end` is not compatible with "
                "taking a `dataloader_iter` argument in your `training_step`."
            )
        if is_overridden("tbptt_split_batch", model):
            raise MisconfigurationException(
                "The model hook `tbptt_split_batch` is not compatible with "
                "taking a `dataloader_iter` argument in your `training_step`."
            )
        if trainer.accumulate_grad_batches != 1:
            raise MisconfigurationException(
                "`accumulate_grad_batches` can only be 1 when your "
                "`training_step` takes `dataloader_iter` as an argument."
            )

        self.trainer = trainer

        # The following field is not used by the processor since it doesn't support automatic
        # optimization and tbptt. Initializing them regardless since they are currently expected by
        # `FitLoop` or `TrainingEpochLoop`.
        # TODO: come up with an abstraction for "batch processors" so they can be better decoupled
        # with parent loops.
        self.accumulated_loss: Optional[torch.Tensor] = None
        self.running_loss: TensorRunningAccum = TensorRunningAccum(window_length=1)
        self.optim_progress = OptimizationProgress()
        self.split_idx: int = 0
        self._skip_backward = False

    def num_active_optimizers(self, batch_idx: Optional[int] = None) -> int:
        """
        Returns the number of active optimizers.
        """
        return len(self.trainer.optimizers)

    def get_active_optimizers(self, batch_idx: Optional[int] = None) -> List[Tuple[int, torch.optim.Optimizer]]:
        """
        Returns the currently active optimizers.

        Returns:
            A list of tuples (opt_idx, optimizer) of currently active optimizers.
        """
        return list(enumerate(self.trainer.optimizers))

    def run(self, dataloader_iter: Iterator) -> Optional[AttributeDict]:
        """
        Args:
            dataloader_iter: the iterator over the dataloader producing the new batch
        """
        batch_idx, (dataloader_iter, is_last) = next(dataloader_iter)

        self.trainer.logger_connector.on_batch_start()
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1)

        self.trainer.fit_loop.epoch_loop.batch_progress.increment_started()

        # give the PL module a result for logging
        model = self.trainer.lightning_module
        # manually capture logged metrics
        model._current_fx_name = "training_step"
        step_kwargs = self._build_kwargs(dataloader_iter, batch_idx)

        with self.trainer.profiler.profile("model_forward"):
            with self.trainer.profiler.profile("training_step"):
                training_step_output = self.trainer.accelerator.training_step(step_kwargs)
                self.trainer.accelerator.post_training_step()

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)
            _check_training_step_output(self.trainer.lightning_module, training_step_output)

            training_step_output, _ = _process_training_step_output(self.trainer, training_step_output)

            if self.trainer.terminate_on_nan:
                check_finite_loss(self.trainer.lightning_module, training_step_output.minimize)

        batch_outputs = [[] for _ in range(len(self.trainer.optimizers))]
        if training_step_output:
            batch_outputs[0].append(training_step_output)

        return AttributeDict(signal=0, training_step_output=batch_outputs, is_last=is_last)

    def teardown(self) -> None:
        """
        No-op. Only defined to comply with FitLoop's expectation.
        """
        pass

    # FIXME: To be deleted in next PR.
    def _build_kwargs(self, dataloader_iter: Iterator, batch_idx: int) -> Dict[str, Any]:
        """Builds the keyword arguments for training_step

        Args:
            dataloader_iter: The dataloader to pass
            batch_idx: the index of the current batch

        Returns:
            An ordered dict with the keyword arguments for the training step
        """
        # enable not needing to add opt_idx to training_step
        step_kwargs = OrderedDict([("dataloader_iter", dataloader_iter)])

        training_step_fx = getattr(self.trainer.lightning_module, "training_step")
        if is_param_in_hook_signature(training_step_fx, "batch_idx"):
            step_kwargs["batch_idx"] = batch_idx

        return step_kwargs

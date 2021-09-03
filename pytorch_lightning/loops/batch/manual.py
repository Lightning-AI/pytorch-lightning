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

from typing import Any, Optional, Tuple

from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.utilities import (
    _build_training_step_kwargs,
    _check_training_step_output,
    _process_training_step_output,
)
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection


class ManualOptimization(Loop):
    @property
    def done(self) -> bool:
        return False

    def reset(self) -> None:
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(
        self, batch: Any, batch_idx: int, hiddens: Optional[Any] = None
    ) -> Optional[Tuple[ResultCollection, Optional[Any]]]:
        """Performs the training step for manual optimization.

        Args:
            batch: the current tbptt split of the current batch
            batch_idx: the index of the current batch
            hiddens: the model's hidden state of the previous iteration

        Returns:
            post-processed outputs from the training step, or ``None`` if training step returned nothing
        """
        # give the PL module a result for logging
        model_ref = self.trainer.lightning_module

        with self.trainer.profiler.profile("model_forward"):

            # TODO: does not need optimizers or opt_idx
            step_kwargs = _build_training_step_kwargs(
                model_ref, self.trainer.optimizers, batch, batch_idx, opt_idx=None, hiddens=hiddens
            )

            # manually capture logged metrics
            model_ref._current_fx_name = "training_step"
            with self.trainer.profiler.profile("training_step"):
                training_step_output = self.trainer.accelerator.training_step(step_kwargs)
                self.trainer.accelerator.post_training_step()

            del step_kwargs

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            _check_training_step_output(self.trainer.lightning_module, training_step_output)

            result_collection, hiddens = _process_training_step_output(self.trainer, training_step_output)
            if result_collection is None:
                return

        return result_collection, hiddens

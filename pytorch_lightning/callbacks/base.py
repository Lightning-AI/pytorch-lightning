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

r"""
Abstract base class used to build new callbacks.

"""
import abc
from typing import TYPE_CHECKING, Any, Dict, List, Union

from torch.optim.optimizer import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer

if TYPE_CHECKING:
    from pytorch_lightning.trainer.trainer import Trainer


class Callback(abc.ABC):
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    def setup(self, trainer: 'Trainer', pl_module: LightningModule, stage: str) -> None:
        """Called when fit or test begins"""

    def teardown(self, trainer: 'Trainer', pl_module: LightningModule, stage: str) -> None:
        """Called when fit or test ends"""

    def on_init_start(self, trainer: 'Trainer') -> None:
        """Called when the trainer: 'Trainer' initialization begins, model has not yet been set."""

    def on_init_end(self, trainer: 'Trainer') -> None:
        """Called when the trainer: 'Trainer' initialization ends, model has not yet been set."""

    def on_fit_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when fit begins"""

    def on_fit_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when fit ends"""

    def on_sanity_check_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the validation sanity check starts."""

    def on_sanity_check_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the validation sanity check ends."""

    def on_train_batch_start(
        self, trainer: 'Trainer', pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(
        self,
        trainer: 'Trainer',
        pl_module: LightningModule,
        outputs: List[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""

    def on_train_epoch_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer: 'Trainer', pl_module: LightningModule, outputs: List[Any]) -> None:
        """Called when the train epoch ends."""

    def on_validation_epoch_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the val epoch begins."""

    def on_validation_epoch_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the val epoch ends."""

    def on_test_epoch_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the test epoch begins."""

    def on_test_epoch_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the test epoch ends."""

    def on_epoch_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the epoch begins."""

    def on_epoch_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the epoch ends."""

    def on_batch_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the training batch begins."""

    def on_validation_batch_start(
        self, trainer: 'Trainer', pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch begins."""

    def on_validation_batch_end(
        self,
        trainer: 'Trainer',
        pl_module: LightningModule,
        outputs: List[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""

    def on_test_batch_start(
        self, trainer: 'Trainer', pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch begins."""

    def on_test_batch_end(
        self,
        trainer: 'Trainer',
        pl_module: LightningModule,
        outputs: List[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""

    def on_batch_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the training batch ends."""

    def on_train_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the train begins."""

    def on_train_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the train ends."""

    def on_pretrain_routine_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the pretrain routine begins."""

    def on_pretrain_routine_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the pretrain routine ends."""

    def on_validation_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""

    def on_validation_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the validation loop ends."""

    def on_test_start(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the test begins."""

    def on_test_end(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the test ends."""

    def on_keyboard_interrupt(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """Called when the training is interrupted by KeyboardInterrupt."""

    def on_save_checkpoint(self, trainer: 'Trainer', pl_module: LightningModule) -> Dict[str, Any]:
        """Called when saving a model checkpoint, use to persist state."""

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        """Called when loading a model checkpoint, use to reload state."""

    def on_after_backward(self, trainer: 'Trainer', pl_module: LightningModule) -> None:
        """
        Called after loss.backward() and before optimizers do anything.
        """

    def on_before_zero_grad(
        self, trainer: 'Trainer', pl_module: LightningModule, optimizer: Union[Optimizer, LightningOptimizer]
    ) -> None:
        """
        Called after optimizer.step() and before optimizer.zero_grad().
        """

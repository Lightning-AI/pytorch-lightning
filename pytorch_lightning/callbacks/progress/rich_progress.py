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
from datetime import timedelta
from typing import Dict, Optional

from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.utilities import _RICH_AVAILABLE

if _RICH_AVAILABLE:
    from rich.console import Console, RenderableType
    from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, TextColumn
    from rich.text import Text

    class CustomTimeColumn(ProgressColumn):

        # Only refresh twice a second to prevent jitter
        max_refresh = 0.5

        def render(self, task) -> Text:
            elapsed = task.finished_time if task.finished else task.elapsed
            remaining = task.time_remaining
            elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
            remaining_delta = "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
            return Text.from_markup(f"[progress.elapsed]{elapsed_delta} < [progress.remaining]{remaining_delta}")

    class BatchesProcessedColumn(ProgressColumn):
        def render(self, task) -> RenderableType:
            return Text.from_markup(f"[magenta] {int(task.completed)}/{task.total}")

    class ProcessingSpeedColumn(ProgressColumn):
        def render(self, task) -> RenderableType:
            task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
            return Text.from_markup(f"[progress.data.speed] {task_speed}it/s")

    class MetricsTextColumn(ProgressColumn):
        """A column containing text."""

        def __init__(self, trainer, stage):
            self._trainer = trainer
            self._stage = stage
            self._tasks = {}
            self._current_task_id = 0
            super().__init__()

        def render(self, task) -> Text:
            if self._stage != "fit" or self._trainer.sanity_checking:
                return ""
            if self._trainer.training and task.id not in self._tasks:
                self._tasks[task.id] = "None"
                if self._renderable_cache:
                    self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
                self._current_task_id = task.id
            if self._trainer.training and task.id != self._current_task_id:
                return self._tasks[task.id]
            _text = ""
            # TODO(@daniellepintz): make this code cleaner
            progress_bar_callback = getattr(self._trainer, "progress_bar_callback", None)
            if progress_bar_callback:
                metrics = self.progress_bar_callback.get_progress_bar_metrics(self.trainer, self)
            else:
                metrics = self._trainer.progress_bar_metrics
            for k, v in metrics.items():
                _text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
            text = Text.from_markup(_text, style=None, justify="left")
            return text


STYLES: Dict[str, str] = {
    "train": "red",
    "sanity_check": "yellow",
    "validate": "yellow",
    "test": "yellow",
    "predict": "yellow",
}


class RichProgressBar(ProgressBarBase):
    """
    Create a progress bar with `rich text formatting <https://github.com/willmcgugan/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichProgressBar

        trainer = Trainer(callbacks=RichProgressBar())

    Args:
        refresh_rate: the number of updates per second, must be strictly positive

    Raises:
        ImportError:
            If required `rich` package is not installed on the device.
    """

    def __init__(self, refresh_rate: float = 1.0):
        if not _RICH_AVAILABLE:
            raise ImportError(
                "`RichProgressBar` requires `rich` to be installed. Install it by running `pip install rich`."
            )
        super().__init__()
        self._refresh_rate: float = refresh_rate
        self._enabled: bool = True
        self._total_val_batches: int = 0
        self.progress: Progress = None
        self.val_sanity_progress_bar_id: Optional[int] = None
        self.main_progress_bar_id: Optional[int] = None
        self.val_progress_bar_id: Optional[int] = None
        self.test_progress_bar_id: Optional[int] = None
        self.predict_progress_bar_id: Optional[int] = None
        self.console = Console(record=True)

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    @property
    def sanity_check_description(self) -> str:
        return "[Validation Sanity Check]"

    @property
    def validation_description(self) -> str:
        return "[Validation]"

    @property
    def test_description(self) -> str:
        return "[Testing]"

    @property
    def predict_description(self) -> str:
        return "[Predicting]"

    def setup(self, trainer, pl_module, stage):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            BatchesProcessedColumn(),
            "[",
            CustomTimeColumn(),
            ProcessingSpeedColumn(),
            MetricsTextColumn(trainer, stage),
            "]",
            console=self.console,
            refresh_per_second=self.refresh_rate,
        ).__enter__()

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)
        self.val_sanity_progress_bar_id = self.progress.add_task(
            f"[{STYLES['sanity_check']}]{self.sanity_check_description}",
            total=trainer.num_sanity_val_steps,
        )

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)
        self.progress.update(self.val_sanity_progress_bar_id, visible=False)

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        self._total_val_batches = self.total_val_batches
        if total_train_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            self._total_val_batches = self._total_val_batches * val_checks_per_epoch

        total_batches = total_train_batches + self._total_val_batches

        train_description = self._get_train_description(trainer.current_epoch)

        self.main_progress_bar_id = self.progress.add_task(
            f"[{STYLES['train']}]{train_description}",
            total=total_batches,
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        if self._total_val_batches > 0:
            self.val_progress_bar_id = self.progress.add_task(
                f"[{STYLES['validate']}]{self.validation_description}",
                total=self._total_val_batches,
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if self.val_progress_bar_id is not None:
            self.progress.update(self.val_progress_bar_id, visible=False)

    def on_test_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.test_progress_bar_id = self.progress.add_task(
            f"[{STYLES['test']}]{self.test_description}",
            total=self.total_test_batches,
        )

    def on_predict_epoch_start(self, trainer, pl_module):
        super().on_predict_epoch_start(trainer, pl_module)
        self.predict_progress_bar_id = self.progress.add_task(
            f"[{STYLES['predict']}]{self.predict_description}",
            total=self.total_predict_batches,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.train_batch_idx, self.total_train_batches + self.total_val_batches):
            self.progress.update(self.main_progress_bar_id, advance=1.0)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if trainer.sanity_checking:
            self.progress.update(self.val_sanity_progress_bar_id, advance=1.0)
        elif self.val_progress_bar_id and self._should_update(
            self.val_batch_idx, self.total_train_batches + self.total_val_batches
        ):
            self.progress.update(self.main_progress_bar_id, advance=1.0)
            self.progress.update(self.val_progress_bar_id, advance=1.0)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.test_batch_idx, self.total_test_batches):
            self.progress.update(self.test_progress_bar_id, advance=1.0)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.predict_batch_idx, self.total_predict_batches):
            self.progress.update(self.predict_progress_bar_id, advance=1.0)

    def _should_update(self, current, total) -> bool:
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    def _get_train_description(self, current_epoch: int) -> str:
        train_description = f"[Epoch {current_epoch}]"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            num_digits = len(str(current_epoch))
            required_padding = (len(self.validation_description) - len(train_description) + 1) - num_digits
            for _ in range(required_padding):
                train_description += " "
        return train_description

    def teardown(self, trainer, pl_module, stage):
        self.progress.__exit__(None, None, None)

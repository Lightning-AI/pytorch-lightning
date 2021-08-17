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

from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.utilities import _RICH_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _RICH_AVAILABLE:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, TextColumn
    from rich.text import Text
else:
    ProgressColumn, TextColumn, Text = None, None, None


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
    def render(self, task) -> Text:
        return Text.from_markup(f"[magenta] {int(task.completed)}/{task.total}")


class ProcessingSpeedColumn(ProgressColumn):
    def render(self, task) -> Text:
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
        if "red" in task.description and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if "red" in task.description and task.id != self._current_task_id:
            return self._tasks[task.id]
        _text = ""
        if self._stage == "test":
            return ""
        if "red" in task.description or "yellow" in task.description:
            for k, v in self._trainer.progress_bar_dict.items():
                _text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
        text = Text.from_markup(_text, style=None, justify="left")
        return text


class RichProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate: int = 1):
        if not _RICH_AVAILABLE:
            raise MisconfigurationException("Rich progress bar is not available")
        super().__init__()
        self._refresh_rate = refresh_rate
        self._enabled = True
        self._total_val_batches = 0
        self.main_progress_bar = None
        self.val_progress_bar = None
        self.test_progress_bar = None
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

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        self._total_val_batches = self.total_val_batches
        if total_train_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            self._total_val_batches = self._total_val_batches * val_checks_per_epoch

        total_batches = total_train_batches + self._total_val_batches
        self.main_progress_bar = self.progress.add_task(
            f"[red][Epoch {trainer.current_epoch}]",
            total=total_batches,
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        if self._total_val_batches > 0:
            self.val_progress_bar = self.progress.add_task(
                "[yellow][Validation]",
                total=self._total_val_batches,
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if self.val_progress_bar is not None:
            self.progress.update(self.val_progress_bar, visible=False)

    def on_test_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        total_test_batches = self.total_test_batches
        self.test_progress_bar = self.progress.add_task(
            "[red][Testing]",
            total=total_test_batches,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.train_batch_idx, self.total_train_batches + self.total_val_batches):
            self.progress.update(self.main_progress_bar, advance=1.0)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.val_progress_bar and self._should_update(
            self.val_batch_idx, self.total_train_batches + self.total_val_batches
        ):
            self.progress.update(self.main_progress_bar, advance=1.0)
            self.progress.update(self.val_progress_bar, advance=1.0)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.test_batch_idx, self.total_test_batches):
            self.progress.update(self.test_progress_bar, advance=1.0)

    def _should_update(self, current, total):
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    def teardown(self, trainer, pl_module, stage):
        self.progress.__exit__(None, None, None)

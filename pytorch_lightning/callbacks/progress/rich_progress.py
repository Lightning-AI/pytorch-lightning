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


class MetricsTextColumn(TextColumn):
    """A column containing text."""

    def __init__(self, trainer):
        self._trainer = trainer
        super().__init__("")

    def render(self, task) -> Text:
        _text = ""
        if "red" in f"{task.description}":
            for k, v in self._trainer.progress_bar_dict.items():
                _text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
        if self.markup:
            text = Text.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        return text


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


class RichProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        if not _RICH_AVAILABLE:
            raise MisconfigurationException("Rich progress bar is not available")
        self._refresh_rate = refresh_rate
        self._process_position = process_position
        self._enabled = True
        self.main_progress_bar = None
        self.val_progress_bar = None
        self.test_progress_bar = None
        self.console = Console(record=True)

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        return self._process_position

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
            MetricsTextColumn(trainer),
            "]",
            console=self.console,
        ).__enter__()

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch

        # total_batches = total_train_batches + total_val_batches
        self.main_progress_bar = self.progress.add_task(
            f"[red][Epoch {trainer.current_epoch}]",
            total=total_train_batches,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.train_batch_idx, self.total_train_batches + self.total_val_batches):
            if getattr(self, "progress", None) is not None:
                self.progress.update(self.main_progress_bar, advance=1.0)
                self.progress.track(trainer.progress_bar_dict)

    def _should_update(self, current, total):
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    def teardown(self, trainer, pl_module, stage):
        self.progress.__exit__(None, None, None)

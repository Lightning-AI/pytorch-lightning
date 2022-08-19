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
import math
import operator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional, Union

from torchmetrics.utilities.imports import _compare_version

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.utilities.imports import _package_available

_RICH_AVAILABLE: bool = _package_available("rich") and _compare_version("rich", operator.ge, "10.2.2")

Task, Style = None, None
if _RICH_AVAILABLE:
    from rich import get_console, reconfigure
    from rich.console import RenderableType
    from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID, TextColumn
    from rich.progress_bar import ProgressBar
    from rich.style import Style
    from rich.text import Text

    class CustomBarColumn(BarColumn):
        """Overrides ``BarColumn`` to provide support for dataloaders that do not define a size (infinite size)
        such as ``IterableDataset``."""

        def render(self, task: "Task") -> ProgressBar:
            """Gets a progress bar widget for a task."""
            return ProgressBar(
                total=max(0, task.total),
                completed=max(0, task.completed),
                width=None if self.bar_width is None else max(1, self.bar_width),
                pulse=not task.started or not math.isfinite(task.remaining),
                animation_time=task.get_time(),
                style=self.style,
                complete_style=self.complete_style,
                finished_style=self.finished_style,
                pulse_style=self.pulse_style,
            )

    @dataclass
    class CustomInfiniteTask(Task):
        """Overrides ``Task`` to define an infinite task.

        This is useful for datasets that do not define a size (infinite size) such as ``IterableDataset``.
        """

        @property
        def time_remaining(self) -> Optional[float]:
            return None

    class CustomProgress(Progress):
        """Overrides ``Progress`` to support adding tasks that have an infinite total size."""

        def add_task(
            self,
            description: str,
            start: bool = True,
            total: float = 100.0,
            completed: int = 0,
            visible: bool = True,
            **fields: Any,
        ) -> TaskID:
            if not math.isfinite(total):
                task = CustomInfiniteTask(
                    self._task_index,
                    description,
                    total,
                    completed,
                    visible=visible,
                    fields=fields,
                    _get_time=self.get_time,
                    _lock=self._lock,
                )
                return self.add_custom_task(task)
            return super().add_task(description, start, total, completed, visible, **fields)

        def add_custom_task(self, task: CustomInfiniteTask, start: bool = True):
            with self._lock:
                self._tasks[self._task_index] = task
                if start:
                    self.start_task(self._task_index)
                new_task_index = self._task_index
                self._task_index = TaskID(int(self._task_index) + 1)
            self.refresh()
            return new_task_index

    class CustomTimeColumn(ProgressColumn):

        # Only refresh twice a second to prevent jitter
        max_refresh = 0.5

        def __init__(self, style: Union[str, Style]) -> None:
            self.style = style
            super().__init__()

        def render(self, task) -> Text:
            elapsed = task.finished_time if task.finished else task.elapsed
            remaining = task.time_remaining
            elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
            remaining_delta = "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
            return Text(f"{elapsed_delta} • {remaining_delta}", style=self.style)

    class BatchesProcessedColumn(ProgressColumn):
        def __init__(self, style: Union[str, Style]):
            self.style = style
            super().__init__()

        def render(self, task) -> RenderableType:
            total = task.total if task.total != float("inf") else "--"
            return Text(f"{int(task.completed)}/{total}", style=self.style)

    class ProcessingSpeedColumn(ProgressColumn):
        def __init__(self, style: Union[str, Style]):
            self.style = style
            super().__init__()

        def render(self, task) -> RenderableType:
            task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
            return Text(f"{task_speed}it/s", style=self.style)

    class MetricsTextColumn(ProgressColumn):
        """A column containing text."""

        def __init__(self, trainer, style):
            self._trainer = trainer
            self._tasks = {}
            self._current_task_id = 0
            self._metrics = {}
            self._style = style
            super().__init__()

        def update(self, metrics):
            # Called when metrics are ready to be rendered.
            # This is to prevent render from causing deadlock issues by requesting metrics
            # in separate threads.
            self._metrics = metrics

        def render(self, task) -> Text:
            if (
                self._trainer.state.fn != "fit"
                or self._trainer.sanity_checking
                or self._trainer.progress_bar_callback.main_progress_bar_id != task.id
            ):
                return Text()
            if self._trainer.training and task.id not in self._tasks:
                self._tasks[task.id] = "None"
                if self._renderable_cache:
                    self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
                self._current_task_id = task.id
            if self._trainer.training and task.id != self._current_task_id:
                return self._tasks[task.id]

            text = ""
            for k, v in self._metrics.items():
                text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
            return Text(text, justify="left", style=self._style)


@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html
    """

    description: Union[str, Style] = "white"
    progress_bar: Union[str, Style] = "#6206E0"
    progress_bar_finished: Union[str, Style] = "#6206E0"
    progress_bar_pulse: Union[str, Style] = "#6206E0"
    batch_progress: Union[str, Style] = "white"
    time: Union[str, Style] = "grey54"
    processing_speed: Union[str, Style] = "grey70"
    metrics: Union[str, Style] = "white"


class RichProgressBar(ProgressBarBase):
    """Create a progress bar with `rich text formatting <https://github.com/Textualize/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichProgressBar

        trainer = Trainer(callbacks=RichProgressBar())

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display.
        leave: Leaves the finished progress bar in the terminal at the end of the epoch. Default: False
        theme: Contains styles used to stylize the progress bar.
        console_kwargs: Args for constructing a `Console`

    Raises:
        ModuleNotFoundError:
            If required `rich` package is not installed on the device.

    Note:
        PyCharm users will need to enable “emulate terminal” in output console option in
        run/debug configuration to see styled output.
        Reference: https://rich.readthedocs.io/en/latest/introduction.html#requirements
    """

    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = RichProgressBarTheme(),
        console_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not _RICH_AVAILABLE:
            raise ModuleNotFoundError(
                "`RichProgressBar` requires `rich` >= 10.2.2. Install it by running `pip install -U rich`."
            )

        super().__init__()
        self._refresh_rate: int = refresh_rate
        self._leave: bool = leave
        self._console_kwargs = console_kwargs or {}
        self._enabled: bool = True
        self.progress: Optional[Progress] = None
        self.val_sanity_progress_bar_id: Optional[int] = None
        self._reset_progress_bar_ids()
        self._metric_component = None
        self._progress_stopped: bool = False
        self.theme = theme
        self._update_for_light_colab_theme()

    @property
    def refresh_rate(self) -> float:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def _update_for_light_colab_theme(self) -> None:
        if _detect_light_colab_theme():
            attributes = ["description", "batch_progress", "metrics"]
            for attr in attributes:
                if getattr(self.theme, attr) == "white":
                    setattr(self.theme, attr, "black")

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(trainer, self.theme.metrics)
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def refresh(self) -> None:
        if self.progress:
            self.progress.refresh()

    def on_train_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_predict_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_test_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_validation_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_sanity_check_start(self, trainer, pl_module):
        self._init_progress(trainer)

    def on_sanity_check_end(self, trainer, pl_module):
        if self.progress is not None:
            self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)
        self.refresh()

    def on_train_epoch_start(self, trainer, pl_module):
        total_batches = self.total_batches_current_epoch
        train_description = self._get_train_description(trainer.current_epoch)

        if self.main_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.main_progress_bar_id is None:
            self.main_progress_bar_id = self._add_task(total_batches, train_description)
        elif self.progress is not None:
            self.progress.reset(
                self.main_progress_bar_id, total=total_batches, description=train_description, visible=True
            )

        self.refresh()

    def on_validation_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        if trainer.sanity_checking:
            if self.val_sanity_progress_bar_id is not None:
                self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)

            self.val_sanity_progress_bar_id = self._add_task(
                self.total_val_batches_current_dataloader, self.sanity_check_description, visible=False
            )
        else:
            if self.val_progress_bar_id is not None:
                self.progress.update(self.val_progress_bar_id, advance=0, visible=False)

            # TODO: remove old tasks when new onces are created
            self.val_progress_bar_id = self._add_task(
                self.total_val_batches_current_dataloader, self.validation_description, visible=False
            )

        self.refresh()

    def _add_task(self, total_batches: int, description: str, visible: bool = True) -> Optional[int]:
        if self.progress is not None:
            return self.progress.add_task(
                f"[{self.theme.description}]{description}", total=total_batches, visible=visible
            )

    def _update(self, progress_bar_id: int, current: int, visible: bool = True) -> None:
        if self.progress is not None and self.is_enabled:
            total = self.progress.tasks[progress_bar_id].total
            if not self._should_update(current, total):
                return

            leftover = current % self.refresh_rate
            advance = leftover if (current == total and leftover != 0) else self.refresh_rate
            self.progress.update(progress_bar_id, advance=advance, visible=visible)
            self.refresh()

    def _should_update(self, current: int, total: Union[int, float]) -> bool:
        return current % self.refresh_rate == 0 or current == total

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_progress_bar_id is not None and trainer.state.fn == "fit":
            self.progress.update(self.val_progress_bar_id, advance=0, visible=False)
            self.refresh()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.state.fn == "fit":
            self._update_metrics(trainer, pl_module)
        self.reset_dataloader_idx_tracker()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.reset_dataloader_idx_tracker()

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.reset_dataloader_idx_tracker()

    def on_test_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        if self.test_progress_bar_id is not None:
            self.progress.update(self.test_progress_bar_id, advance=0, visible=False)
        self.test_progress_bar_id = self._add_task(self.total_test_batches_current_dataloader, self.test_description)
        self.refresh()

    def on_predict_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        if self.predict_progress_bar_id is not None:
            self.progress.update(self.predict_progress_bar_id, advance=0, visible=False)
        self.predict_progress_bar_id = self._add_task(
            self.total_predict_batches_current_dataloader, self.predict_description
        )
        self.refresh()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._update(self.main_progress_bar_id, self.train_batch_idx + self._val_processed)
        self._update_metrics(trainer, pl_module)
        self.refresh()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._update_metrics(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.sanity_checking:
            self._update(self.val_sanity_progress_bar_id, self.val_batch_idx)
        elif self.val_progress_bar_id is not None:
            # check to see if we should update the main training progress bar
            if self.main_progress_bar_id is not None:
                self._update(self.main_progress_bar_id, self.train_batch_idx + self._val_processed)
            self._update(self.val_progress_bar_id, self.val_batch_idx)
        self.refresh()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._update(self.test_progress_bar_id, self.test_batch_idx)
        self.refresh()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._update(self.predict_progress_bar_id, self.predict_batch_idx)
        self.refresh()

    def _get_train_description(self, current_epoch: int) -> str:
        train_description = f"Epoch {current_epoch}"
        if self.trainer.max_epochs is not None:
            train_description += f"/{self.trainer.max_epochs - 1}"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            train_description = f"{train_description:{len(self.validation_description)}}"
        return train_description

    def _stop_progress(self) -> None:
        if self.progress is not None:
            self.progress.stop()
            # # signals for progress to be re-initialized for next stages
            self._progress_stopped = True

    def _reset_progress_bar_ids(self):
        self.main_progress_bar_id: Optional[int] = None
        self.val_progress_bar_id: Optional[int] = None
        self.test_progress_bar_id: Optional[int] = None
        self.predict_progress_bar_id: Optional[int] = None

    def _update_metrics(self, trainer, pl_module) -> None:
        metrics = self.get_metrics(trainer, pl_module)
        if self._metric_component:
            self._metric_component.update(metrics)

    def teardown(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        self._stop_progress()

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self._stop_progress()

    @property
    def val_progress_bar(self) -> Task:
        return self.progress.tasks[self.val_progress_bar_id]

    @property
    def val_sanity_check_bar(self) -> Task:
        return self.progress.tasks[self.val_sanity_progress_bar_id]

    @property
    def main_progress_bar(self) -> Task:
        return self.progress.tasks[self.main_progress_bar_id]

    @property
    def test_progress_bar(self) -> Task:
        return self.progress.tasks[self.test_progress_bar_id]

    def configure_columns(self, trainer) -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]


def _detect_light_colab_theme() -> bool:
    """Detect if it's light theme in Colab."""
    try:
        get_ipython  # type: ignore
    except NameError:
        return False
    ipython = get_ipython()  # noqa: F821
    if "google.colab" in str(ipython.__class__):
        try:
            from google.colab import output

            return output.eval_js('document.documentElement.matches("[theme=light]")')
        except ModuleNotFoundError:
            return False
    return False

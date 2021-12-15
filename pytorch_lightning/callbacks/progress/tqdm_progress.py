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
import importlib
import io
import math
import os
import sys
from typing import Optional, Union

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.base import ProgressBarBase

_PAD_SIZE = 5


class Tqdm(_tqdm):
    def __init__(self, *args, **kwargs):
        """Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from
        flickering."""
        # this just to make the make docs happy, otherwise it pulls docs which has some issues...
        super().__init__(*args, **kwargs)

    @staticmethod
    def format_num(n) -> str:
        """Add additional padding to the formatted numbers."""
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
        if should_be_padded and "e" not in n:
            if "." not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += "."
            n += "0" * (_PAD_SIZE - len(n))
        return n


class TQDMProgressBar(ProgressBarBase):
    r"""
    This is the default progress bar used by Lightning. It prints to ``stdout`` using the
    :mod:`tqdm` package and shows up to four different bars:

        - **sanity check progress:** the progress during the sanity check run
        - **main progress:** shows training + validation progress combined. It also accounts for
          multiple validation runs during training when
          :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval` is used.
        - **validation progress:** only visible during validation;
          shows total progress over all validation datasets.
        - **test progress:** only active when testing; shows total progress over all test datasets.

    For infinite datasets, the progress bar never ends.

    If you want to customize the default ``tqdm`` progress bars used by Lightning, you can override
    specific methods of the callback class and pass your custom implementation to the
    :class:`~pytorch_lightning.trainer.trainer.Trainer`.

    Example:

        >>> class LitProgressBar(TQDMProgressBar):
        ...     def init_validation_tqdm(self):
        ...         bar = super().init_validation_tqdm()
        ...         bar.set_description('running validation ...')
        ...         return bar
        ...
        >>> bar = LitProgressBar()
        >>> from pytorch_lightning import Trainer
        >>> trainer = Trainer(callbacks=[bar])

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display. By default, the :class:`~pytorch_lightning.trainer.trainer.Trainer`
            uses this implementation of the progress bar and sets the refresh rate to the value provided to the
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.progress_bar_refresh_rate` argument in the
            :class:`~pytorch_lightning.trainer.trainer.Trainer`.
        process_position: Set this to a value greater than ``0`` to offset the progress bars by this many lines.
            This is useful when you have progress bars defined elsewhere and want to show all of them
            together. This corresponds to
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.process_position` in the
            :class:`~pytorch_lightning.trainer.trainer.Trainer`.
    """

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__()
        self._refresh_rate = refresh_rate
        self._process_position = process_position
        self._enabled = True
        self.main_progress_bar = None
        self.val_progress_bar = None
        self.test_progress_bar = None
        self.predict_progress_bar = None

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = self.__dict__.copy()
        state["main_progress_bar"] = None
        state["val_progress_bar"] = None
        state["test_progress_bar"] = None
        state["predict_progress_bar"] = None
        return state

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

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        bar = Tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = Tqdm(
            desc="Predicting",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = Tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def on_sanity_check_start(self, trainer, pl_module):
        self.val_progress_bar = self.init_sanity_tqdm()
        self.main_progress_bar = Tqdm(disable=True)  # dummy progress bar

    def on_sanity_check_end(self, trainer, pl_module):
        self.main_progress_bar.close()
        self.val_progress_bar.close()

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = self.init_train_tqdm()

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float("inf") and total_val_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_batches = total_train_batches + total_val_batches
        reset(self.main_progress_bar, total=total_batches, current=self.train_batch_idx)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self._should_update(self.train_batch_idx):
            self._update_bar(self.main_progress_bar)
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled:
            self._update_bar(self.main_progress_bar)
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.main_progress_bar.close()

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        if trainer.sanity_checking:
            reset(self.val_progress_bar, total=sum(trainer.num_sanity_val_batches), current=self.val_batch_idx)
        else:
            if trainer.state.fn == pl.trainer.states.TrainerFn.FITTING:
                self._update_bar(self.main_progress_bar)  # fill up remaining
            self.val_progress_bar = self.init_validation_tqdm()
            reset(self.val_progress_bar, total=self.total_val_batches, current=self.val_batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.val_batch_idx):
            self._update_bar(self.val_progress_bar)
            if trainer.state.fn == pl.trainer.states.TrainerFn.FITTING:
                self._update_bar(self.main_progress_bar)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled:
            self._update_bar(self.val_progress_bar)

    def on_validation_end(self, trainer, pl_module):
        if self.main_progress_bar is not None and trainer.state.fn == pl.trainer.states.TrainerFn.FITTING:
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        self.val_progress_bar.close()

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self.test_progress_bar = self.init_test_tqdm()
        self.test_progress_bar.total = convert_inf(self.total_test_batches)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.test_batch_idx):
            self._update_bar(self.test_progress_bar)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_enabled:
            self._update_bar(self.test_progress_bar)

    def on_test_end(self, trainer, pl_module):
        self.test_progress_bar.close()

    def on_predict_epoch_start(self, trainer, pl_module):
        super().on_predict_epoch_start(trainer, pl_module)
        self.predict_progress_bar = self.init_predict_tqdm()
        self.predict_progress_bar.total = convert_inf(self.total_predict_batches)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.predict_batch_idx):
            self._update_bar(self.predict_progress_bar)

    def on_predict_end(self, trainer, pl_module):
        self.predict_progress_bar.close()

    def print(
        self, *args, sep: str = " ", end: str = os.linesep, file: Optional[io.TextIOBase] = None, nolock: bool = False
    ):
        active_progress_bar = None

        if self.main_progress_bar is not None and not self.main_progress_bar.disable:
            active_progress_bar = self.main_progress_bar
        elif self.val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self.test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self.predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, end=end, file=file, nolock=nolock)

    def _should_update(self, idx: int) -> bool:
        return self.is_enabled and (idx % self.refresh_rate == 0)

    def _update_bar(self, bar: Optional[Tqdm]) -> None:
        """Updates the bar by the refresh rate without overshooting."""
        if bar is None:
            return
        if bar.total is not None:
            delta = min(self.refresh_rate, bar.total - bar.n)
        else:
            # infinite / unknown size
            delta = self.refresh_rate
        if delta > 0:
            bar.update(delta)


def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.

    We have to convert it to None.
    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x


def reset(bar: Tqdm, total: Optional[int] = None, current: int = 0) -> None:
    """Resets the tqdm bar to the desired position and sets a new total, unless it is disabled."""
    if not bar.disable:
        bar.reset(total=convert_inf(total))
        bar.n = current

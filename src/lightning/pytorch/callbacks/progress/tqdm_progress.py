# Copyright The Lightning AI team.
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
import math
import os
import sys
from typing import Any, Optional, Union

from typing_extensions import override

from lightning.pytorch.utilities.types import STEP_OUTPUT

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_debug

_PAD_SIZE = 5


class Tqdm(_tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from
        flickering."""
        # this just to make the make docs happy, otherwise it pulls docs which has some issues...
        super().__init__(*args, **kwargs)

    @staticmethod
    def format_num(n: Union[int, float, str]) -> str:
        """Add additional padding to the formatted numbers."""
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
            assert isinstance(n, str)
        if should_be_padded and "e" not in n:
            if "." not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += "."
            n += "0" * (_PAD_SIZE - len(n))
        return n


class TQDMProgressBar(ProgressBar):
    r"""This is the default progress bar used by Lightning. It prints to ``stdout`` using the :mod:`tqdm` package and
    shows up to four different bars:

        - **sanity check progress:** the progress during the sanity check run
        - **train progress:** shows the training progress. It will pause if validation starts and will resume
          when it ends, and also accounts for multiple validation runs during training when
          :paramref:`~lightning.pytorch.trainer.trainer.Trainer.val_check_interval` is used.
        - **validation progress:** only visible during validation;
          shows total progress over all validation datasets.
        - **test progress:** only active when testing; shows total progress over all test datasets.

    For infinite datasets, the progress bar never ends.

    If you want to customize the default ``tqdm`` progress bars used by Lightning, you can override
    specific methods of the callback class and pass your custom implementation to the
    :class:`~lightning.pytorch.trainer.trainer.Trainer`.

    Example:

        >>> class LitProgressBar(TQDMProgressBar):
        ...     def init_validation_tqdm(self):
        ...         bar = super().init_validation_tqdm()
        ...         bar.set_description('running validation ...')
        ...         return bar
        ...
        >>> bar = LitProgressBar()
        >>> from lightning.pytorch import Trainer
        >>> trainer = Trainer(callbacks=[bar])

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display.
        process_position: Set this to a value greater than ``0`` to offset the progress bars by this many lines.
            This is useful when you have progress bars defined elsewhere and want to show all of them
            together.
        leave: If set to ``True``, leaves the finished progress bar in the terminal at the end of the epoch.
            Default: ``False``

    """

    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"

    def __init__(self, refresh_rate: int = 1, process_position: int = 0, leave: bool = False):
        super().__init__()
        self._refresh_rate = self._resolve_refresh_rate(refresh_rate)
        self._process_position = process_position
        self._enabled = True
        self._train_progress_bar: Optional[_tqdm] = None
        self._val_progress_bar: Optional[_tqdm] = None
        self._test_progress_bar: Optional[_tqdm] = None
        self._predict_progress_bar: Optional[_tqdm] = None
        self._leave = leave

    def __getstate__(self) -> dict:
        # can't pickle the tqdm objects
        return {k: v if not isinstance(v, _tqdm) else None for k, v in vars(self).items()}

    @property
    def train_progress_bar(self) -> _tqdm:
        if self._train_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._train_progress_bar` reference has not been set yet.")
        return self._train_progress_bar

    @train_progress_bar.setter
    def train_progress_bar(self, bar: _tqdm) -> None:
        self._train_progress_bar = bar

    @property
    def val_progress_bar(self) -> _tqdm:
        if self._val_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._val_progress_bar` reference has not been set yet.")
        return self._val_progress_bar

    @val_progress_bar.setter
    def val_progress_bar(self, bar: _tqdm) -> None:
        self._val_progress_bar = bar

    @property
    def test_progress_bar(self) -> _tqdm:
        if self._test_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._test_progress_bar` reference has not been set yet.")
        return self._test_progress_bar

    @test_progress_bar.setter
    def test_progress_bar(self, bar: _tqdm) -> None:
        self._test_progress_bar = bar

    @property
    def predict_progress_bar(self) -> _tqdm:
        if self._predict_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._predict_progress_bar` reference has not been set yet.")
        return self._predict_progress_bar

    @predict_progress_bar.setter
    def predict_progress_bar(self, bar: _tqdm) -> None:
        self._predict_progress_bar = bar

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

    @override
    def disable(self) -> None:
        self._enabled = False

    @override
    def enable(self) -> None:
        self._enabled = True

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        return Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        return Tqdm(
            desc=self.predict_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        return Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    @override
    def on_sanity_check_start(self, *_: Any) -> None:
        self.val_progress_bar = self.init_sanity_tqdm()
        self.train_progress_bar = Tqdm(disable=True)  # dummy progress bar

    @override
    def on_sanity_check_end(self, *_: Any) -> None:
        self.val_progress_bar.close()
        self.train_progress_bar.close()

    @override
    def on_train_start(self, *_: Any) -> None:
        self.train_progress_bar = self.init_train_tqdm()

    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.reset(convert_inf(self.total_train_batches))
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    @override
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        if self._leave:
            self.train_progress_bar.close()

    @override
    def on_train_end(self, *_: Any) -> None:
        self.train_progress_bar.close()

    @override
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            self.val_progress_bar = self.init_validation_tqdm()

    @override
    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.val_progress_bar.reset(convert_inf(self.total_val_batches_current_dataloader))
        self.val_progress_bar.initial = 0
        desc = self.sanity_check_description if trainer.sanity_checking else self.validation_description
        self.val_progress_bar.set_description(f"{desc} DataLoader {dataloader_idx}")

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()
        if self._train_progress_bar is not None and trainer.state.fn == "fit":
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    @override
    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_progress_bar = self.init_test_tqdm()

    @override
    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.test_progress_bar.reset(convert_inf(self.total_test_batches_current_dataloader))
        self.test_progress_bar.initial = 0
        self.test_progress_bar.set_description(f"{self.test_description} DataLoader {dataloader_idx}")

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.test_progress_bar.total):
            _update_n(self.test_progress_bar, n)

    @override
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_progress_bar.close()
        self.reset_dataloader_idx_tracker()

    @override
    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.predict_progress_bar = self.init_predict_tqdm()

    @override
    def on_predict_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.predict_progress_bar.reset(convert_inf(self.total_predict_batches_current_dataloader))
        self.predict_progress_bar.initial = 0
        self.predict_progress_bar.set_description(f"{self.predict_description} DataLoader {dataloader_idx}")

    @override
    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.predict_progress_bar.total):
            _update_n(self.predict_progress_bar, n)

    @override
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.predict_progress_bar.close()
        self.reset_dataloader_idx_tracker()

    @override
    def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
        active_progress_bar = None

        if self._train_progress_bar is not None and not self.train_progress_bar.disable:
            active_progress_bar = self.train_progress_bar
        elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, **kwargs)

    def _should_update(self, current: int, total: int) -> bool:
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        if os.getenv("COLAB_GPU") and refresh_rate == 1:
            # smaller refresh rate on colab causes crashes, choose a higher value
            rank_zero_debug("Using a higher refresh rate on Colab. Setting it to `20`")
            return 20
        # Support TQDM_MINITERS environment variable, which sets the minimum refresh rate
        if "TQDM_MINITERS" in os.environ:
            return max(int(os.environ["TQDM_MINITERS"]), refresh_rate)
        return refresh_rate


def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.

    We have to convert it to None.

    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x


def _update_n(bar: _tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()

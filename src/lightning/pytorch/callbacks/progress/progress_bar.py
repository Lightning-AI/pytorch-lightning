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
from typing import Any, Dict, Optional, Union

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class ProgressBar(Callback):
    r"""The base class for progress bars in Lightning. It is a :class:`~lightning.pytorch.callbacks.Callback` that keeps
    track of the batch progress in the :class:`~lightning.pytorch.trainer.trainer.Trainer`. You should implement your
    highly custom progress bars with this as the base class.

    Example::

        class LitProgressBar(ProgressBar):

            def __init__(self):
                super().__init__()  # don't forget this :)
                self.enable = True

            def disable(self):
                self.enable = False

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
                percent = (batch_idx / self.total_train_batches) * 100
                sys.stdout.flush()
                sys.stdout.write(f'{percent:.01f} percent complete \r')

        bar = LitProgressBar()
        trainer = Trainer(callbacks=[bar])

    """

    def __init__(self) -> None:
        self._trainer: Optional["pl.Trainer"] = None
        self._current_eval_dataloader_idx: Optional[int] = None

    @property
    def trainer(self) -> "pl.Trainer":
        if self._trainer is None:
            raise TypeError(f"The `{self.__class__.__name__}._trainer` reference has not been set yet.")
        return self._trainer

    @property
    def sanity_check_description(self) -> str:
        return "Sanity Checking"

    @property
    def train_description(self) -> str:
        return "Training"

    @property
    def validation_description(self) -> str:
        return "Validation"

    @property
    def test_description(self) -> str:
        return "Testing"

    @property
    def predict_description(self) -> str:
        return "Predicting"

    @property
    def total_train_batches(self) -> Union[int, float]:
        """The total number of training batches, which may change from epoch to epoch.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the training
        dataloader is of infinite size.

        """
        return self.trainer.num_training_batches

    @property
    def total_val_batches_current_dataloader(self) -> Union[int, float]:
        """The total number of validation batches, which may change from epoch to epoch for current dataloader.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the validation
        dataloader is of infinite size.

        """
        batches = self.trainer.num_sanity_val_batches if self.trainer.sanity_checking else self.trainer.num_val_batches
        if isinstance(batches, list):
            assert self._current_eval_dataloader_idx is not None
            return batches[self._current_eval_dataloader_idx]
        return batches

    @property
    def total_test_batches_current_dataloader(self) -> Union[int, float]:
        """The total number of testing batches, which may change from epoch to epoch for current dataloader.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the test dataloader is
        of infinite size.

        """
        batches = self.trainer.num_test_batches
        if isinstance(batches, list):
            assert self._current_eval_dataloader_idx is not None
            return batches[self._current_eval_dataloader_idx]
        return batches

    @property
    def total_predict_batches_current_dataloader(self) -> Union[int, float]:
        """The total number of prediction batches, which may change from epoch to epoch for current dataloader.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the predict dataloader
        is of infinite size.

        """
        assert self._current_eval_dataloader_idx is not None
        return self.trainer.num_predict_batches[self._current_eval_dataloader_idx]

    @property
    def total_val_batches(self) -> Union[int, float]:
        """The total number of validation batches, which may change from epoch to epoch for all val dataloaders.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the predict dataloader
        is of infinite size.

        """
        if not self.trainer.fit_loop.epoch_loop._should_check_val_epoch():
            return 0
        return (
            sum(self.trainer.num_val_batches)
            if isinstance(self.trainer.num_val_batches, list)
            else self.trainer.num_val_batches
        )

    def has_dataloader_changed(self, dataloader_idx: int) -> bool:
        old_dataloader_idx = self._current_eval_dataloader_idx
        self._current_eval_dataloader_idx = dataloader_idx
        return old_dataloader_idx != dataloader_idx

    def reset_dataloader_idx_tracker(self) -> None:
        self._current_eval_dataloader_idx = None

    def disable(self) -> None:
        """You should provide a way to disable the progress bar."""
        raise NotImplementedError

    def enable(self) -> None:
        """You should provide a way to enable the progress bar.

        The :class:`~lightning.pytorch.trainer.trainer.Trainer` will call this in e.g. pre-training
        routines like the :ref:`learning rate finder <advanced/training_tricks:Learning Rate Finder>`.
        to temporarily enable and disable the training progress bar.

        """
        raise NotImplementedError

    def print(self, *args: Any, **kwargs: Any) -> None:
        """You should provide a way to print without breaking the progress bar."""
        print(*args, **kwargs)

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._trainer = trainer
        if not trainer.is_global_zero:
            self.disable()

    def get_metrics(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        r"""Combines progress bar metrics collected from the trainer with standard metrics from get_standard_metrics.
        Implement this to override the items displayed in the progress bar.

        Here is an example of how to override the defaults:

        .. code-block:: python

            def get_metrics(self, trainer, model):
                # don't show the version number
                items = super().get_metrics(trainer, model)
                items.pop("v_num", None)
                return items

        Return:
            Dictionary with the items to be displayed in the progress bar.

        """
        standard_metrics = get_standard_metrics(trainer)
        pbar_metrics = trainer.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
            rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                " If this is undesired, change the name or override `get_metrics()` in the progress bar callback.",
            )

        return {**standard_metrics, **pbar_metrics}


def get_standard_metrics(trainer: "pl.Trainer") -> Dict[str, Union[int, str]]:
    r"""Returns the standard metrics displayed in the progress bar. Currently, it only includes the version of the
    experiment when using a logger.

    .. code-block::

        Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, v_num=10]

    Return:
        Dictionary with the standard metrics to be displayed in the progress bar.

    """
    items_dict: Dict[str, Union[int, str]] = {}
    if trainer.loggers:
        from lightning.pytorch.loggers.utilities import _version

        if (version := _version(trainer.loggers)) not in ("", None):
            if isinstance(version, str):
                # show last 4 places of long version strings
                version = version[-4:]
            items_dict["v_num"] = version

    return items_dict

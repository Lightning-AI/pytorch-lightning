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
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.logger import _version
from pytorch_lightning.utilities.rank_zero import rank_zero_warn


class ProgressBarBase(Callback):
    r"""
    The base class for progress bars in Lightning. It is a :class:`~pytorch_lightning.callbacks.Callback`
    that keeps track of the batch progress in the :class:`~pytorch_lightning.trainer.trainer.Trainer`.
    You should implement your highly custom progress bars with this as the base class.

    Example::

        class LitProgressBar(ProgressBarBase):

            def __init__(self):
                super().__init__()  # don't forget this :)
                self.enable = True

            def disable(self):
                self.enable = False

            def on_train_batch_end(self, trainer, pl_module, outputs, batch_idx):
                super().on_train_batch_end(trainer, pl_module, outputs, batch_idx)  # don't forget this :)
                percent = (self.train_batch_idx / self.total_train_batches) * 100
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
    def _val_processed(self) -> int:
        # use total in case validation runs more than once per training epoch
        return self.trainer.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.total.processed

    @property
    def train_batch_idx(self) -> int:
        """The number of batches processed during training.

        Use this to update your progress bar.
        """
        return self.trainer.fit_loop.epoch_loop.batch_progress.current.processed

    @property
    def val_batch_idx(self) -> int:
        """The number of batches processed during validation.

        Use this to update your progress bar.
        """
        if self.trainer.state.fn == "fit":
            loop = self.trainer.fit_loop.epoch_loop.val_loop
        else:
            loop = self.trainer.validate_loop

        current_batch_idx = loop.epoch_loop.batch_progress.current.processed
        return current_batch_idx

    @property
    def test_batch_idx(self) -> int:
        """The number of batches processed during testing.

        Use this to update your progress bar.
        """
        return self.trainer.test_loop.epoch_loop.batch_progress.current.processed

    @property
    def predict_batch_idx(self) -> int:
        """The number of batches processed during prediction.

        Use this to update your progress bar.
        """
        return self.trainer.predict_loop.epoch_loop.batch_progress.current.processed

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
        assert self._current_eval_dataloader_idx is not None
        if self.trainer.sanity_checking:
            return self.trainer.num_sanity_val_batches[self._current_eval_dataloader_idx]

        return self.trainer.num_val_batches[self._current_eval_dataloader_idx]

    @property
    def total_test_batches_current_dataloader(self) -> Union[int, float]:
        """The total number of testing batches, which may change from epoch to epoch for current dataloader.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the test dataloader is
        of infinite size.
        """
        assert self._current_eval_dataloader_idx is not None
        return self.trainer.num_test_batches[self._current_eval_dataloader_idx]

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
        return sum(self.trainer.num_val_batches) if self.trainer.fit_loop.epoch_loop._should_check_val_epoch() else 0

    @property
    def total_batches_current_epoch(self) -> Union[int, float]:
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        assert self._trainer is not None

        if total_train_batches != float("inf") and total_val_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_check_batch = self.trainer.val_check_batch
            if self.trainer.check_val_every_n_epoch is None:
                train_batches_processed = self.trainer.fit_loop.total_batch_idx + 1
                val_checks_per_epoch = ((train_batches_processed + total_train_batches) // val_check_batch) - (
                    train_batches_processed // val_check_batch
                )
            else:
                val_checks_per_epoch = total_train_batches // val_check_batch

            total_val_batches = total_val_batches * val_checks_per_epoch

        return total_train_batches + total_val_batches

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

        The :class:`~pytorch_lightning.trainer.trainer.Trainer` will call this in e.g. pre-training
        routines like the :ref:`learning rate finder <advanced/training_tricks:Learning Rate Finder>`.
        to temporarily enable and disable the main progress bar.
        """
        raise NotImplementedError

    def print(self, *args: Any, **kwargs: Any) -> None:
        """You should provide a way to print without breaking the progress bar."""
        print(*args, **kwargs)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        self._trainer = trainer
        if not trainer.is_global_zero:
            self.disable()

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        r"""
        Combines progress bar metrics collected from the trainer with standard metrics from get_standard_metrics.
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
        standard_metrics = get_standard_metrics(trainer, pl_module)
        pbar_metrics = trainer.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
            rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                " If this is undesired, change the name or override `get_metrics()` in the progress bar callback.",
            )

        return {**standard_metrics, **pbar_metrics}


def get_standard_metrics(trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
    r"""
    Returns several standard metrics displayed in the progress bar, including the average loss value,
    split index of BPTT (if used) and the version of the experiment when using a logger.

    .. code-block::

        Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, loss=4.501, v_num=10]

    Return:
        Dictionary with the standard metrics to be displayed in the progress bar.
    """
    # call .item() only once but store elements without graphs
    running_train_loss = trainer.fit_loop.running_loss.mean()
    avg_training_loss = None
    if running_train_loss is not None:
        avg_training_loss = running_train_loss.cpu().item()
    elif pl_module.automatic_optimization:
        avg_training_loss = float("NaN")

    items_dict: Dict[str, Union[int, str]] = {}
    if avg_training_loss is not None:
        items_dict["loss"] = f"{avg_training_loss:.3g}"

    if pl_module.truncated_bptt_steps > 0:
        items_dict["split_idx"] = trainer.fit_loop.split_idx

    if trainer.loggers:
        version = _version(trainer.loggers)
        if version is not None:
            if isinstance(version, str):
                # show last 4 places of long version strings
                version = version[-4:]
            items_dict["v_num"] = version

    return items_dict

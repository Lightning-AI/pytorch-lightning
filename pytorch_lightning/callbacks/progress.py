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

"""
Progress Bars
=============

Use or override one of the progress bar callbacks.

"""
import importlib
import sys


# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

from pytorch_lightning.callbacks import Callback


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

            def on_train_batch_end(self, trainer, pl_module, outputs):
                super().on_train_batch_end(trainer, pl_module, outputs)  # don't forget this :)
                percent = (self.train_batch_idx / self.total_train_batches) * 100
                sys.stdout.flush()
                sys.stdout.write(f'{percent:.01f} percent complete \r')

        bar = LitProgressBar()
        trainer = Trainer(callbacks=[bar])

    """
    def __init__(self):

        self._trainer = None
        self._train_batch_idx = 0
        self._val_batch_idx = 0
        self._test_batch_idx = 0

    @property
    def trainer(self):
        return self._trainer

    @property
    def train_batch_idx(self) -> int:
        """
        The current batch index being processed during training.
        Use this to update your progress bar.
        """
        return self._train_batch_idx

    @property
    def val_batch_idx(self) -> int:
        """
        The current batch index being processed during validation.
        Use this to update your progress bar.
        """
        return self._val_batch_idx

    @property
    def test_batch_idx(self) -> int:
        """
        The current batch index being processed during testing.
        Use this to update your progress bar.
        """
        return self._test_batch_idx

    @property
    def total_train_batches(self) -> int:
        """
        The total number of training batches during training, which may change from epoch to epoch.
        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the
        training dataloader is of infinite size.
        """
        return self.trainer.num_training_batches

    @property
    def total_val_batches(self) -> int:
        """
        The total number of training batches during validation, which may change from epoch to epoch.
        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the
        validation dataloader is of infinite size.
        """
        total_val_batches = 0
        if not self.trainer.disable_validation:
            is_val_epoch = (self.trainer.current_epoch) % self.trainer.check_val_every_n_epoch == 0
            total_val_batches = sum(self.trainer.num_val_batches) if is_val_epoch else 0
        return total_val_batches

    @property
    def total_test_batches(self) -> int:
        """
        The total number of training batches during testing, which may change from epoch to epoch.
        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the
        test dataloader is of infinite size.
        """
        return sum(self.trainer.num_test_batches)

    def disable(self):
        """
        You should provide a way to disable the progress bar.
        The :class:`~pytorch_lightning.trainer.trainer.Trainer` will call this to disable the
        output on processes that have a rank different from 0, e.g., in multi-node training.
        """
        raise NotImplementedError

    def enable(self):
        """
        You should provide a way to enable the progress bar.
        The :class:`~pytorch_lightning.trainer.trainer.Trainer` will call this in e.g. pre-training
        routines like the :ref:`learning rate finder <lr_finder>` to temporarily enable and
        disable the main progress bar.
        """
        raise NotImplementedError

    def on_init_end(self, trainer):
        self._trainer = trainer

    def on_train_start(self, trainer, pl_module):
        self._train_batch_idx = trainer.batch_idx

    def on_epoch_start(self, trainer, pl_module):
        self._train_batch_idx = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._train_batch_idx += 1

    def on_validation_start(self, trainer, pl_module):
        self._val_batch_idx = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._val_batch_idx += 1

    def on_test_start(self, trainer, pl_module):
        self._test_batch_idx = 0

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._test_batch_idx += 1


class ProgressBar(ProgressBarBase):
    r"""
    This is the default progress bar used by Lightning. It prints to `stdout` using the
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
    :class:`~pytorch_lightning.trainer.trainer.Trainer`:

    Example::

        class LitProgressBar(ProgressBar):

            def init_validation_tqdm(self):
                bar = super().init_validation_tqdm()
                bar.set_description('running validation ...')
                return bar

        bar = LitProgressBar()
        trainer = Trainer(callbacks=[bar])

    Args:
        refresh_rate:
            Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display. By default, the
            :class:`~pytorch_lightning.trainer.trainer.Trainer` uses this implementation of the progress
            bar and sets the refresh rate to the value provided to the
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.progress_bar_refresh_rate` argument in the
            :class:`~pytorch_lightning.trainer.trainer.Trainer`.
        process_position:
            Set this to a value greater than ``0`` to offset the progress bars by this many lines.
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

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = self.__dict__.copy()
        state['main_progress_bar'] = None
        state['val_progress_bar'] = None
        state['test_progress_bar'] = None
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

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc='Validation sanity check',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc='Testing',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)
        self.val_progress_bar = self.init_sanity_tqdm()
        self.val_progress_bar.total = convert_inf(sum(trainer.num_sanity_val_batches))
        self.main_progress_bar = tqdm(disable=True)  # dummy progress bar

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)
        self.main_progress_bar.close()
        self.val_progress_bar.close()

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = self.init_train_tqdm()

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float('inf') and not trainer.fast_dev_run:
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_batches = total_train_batches + total_val_batches
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset(convert_inf(total_batches))
        self.main_progress_bar.set_description(f'Epoch {trainer.current_epoch}')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
            self.main_progress_bar.update(self.refresh_rate)
            self.main_progress_bar.set_postfix(trainer.progress_bar_dict)

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        if not trainer.running_sanity_check:
            self.val_progress_bar = self.init_validation_tqdm()
            self.val_progress_bar.total = convert_inf(self.total_val_batches)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.val_batch_idx % self.refresh_rate == 0:
            self.val_progress_bar.update(self.refresh_rate)
            self.main_progress_bar.update(self.refresh_rate)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        self.main_progress_bar.set_postfix(trainer.progress_bar_dict)
        self.val_progress_bar.close()

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.main_progress_bar.close()

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self.test_progress_bar = self.init_test_tqdm()
        self.test_progress_bar.total = convert_inf(self.total_test_batches)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.is_enabled and self.test_batch_idx % self.refresh_rate == 0:
            self.test_progress_bar.update(self.refresh_rate)

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        self.test_progress_bar.close()


def convert_inf(x):
    """ The tqdm doesn't support inf values. We have to convert it to None. """
    if x == float('inf'):
        return None
    return x

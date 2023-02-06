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
r"""
BatchSizeFinder
===============

Finds optimal batch size
"""

from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.tuner.batch_size_scaling import scale_batch_size
from pytorch_lightning.utilities.exceptions import _TunerExitException, MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn


class BatchSizeFinder(Callback):
    """The ``BatchSizeFinder`` callback tries to find the largest batch size for a given model that does not give
    an out of memory (OOM) error. All you need to do is add it as a callback inside Trainer and call
    ``trainer.{fit,validate,test,predict}``. Internally it calls the respective step function ``steps_per_trial``
    times for each batch size until one of the batch sizes generates an OOM error.

    Args:
        mode: search strategy to update the batch size:

            - ``'power'``: Keep multiplying the batch size by 2, until we get an OOM error.
            - ``'binsearch'``: Initially keep multiplying by 2 and after encountering an OOM error
                do a binary search between the last successful batch size and the batch size that failed.

        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs,
            however in practice a few are needed.

        init_val: initial batch size to start the search with.

        max_trials: max number of increases in batch size done before
            algorithm is terminated

        batch_arg_name: name of the attribute that stores the batch size.
            It is expected that the user has provided a model or datamodule that has a hyperparameter
            with that name. We will look for this attribute name in the following places

            - ``model``
            - ``model.hparams``
            - ``trainer.datamodule`` (the datamodule passed to the tune method)

    Example::

        # 1. Customize the BatchSizeFinder callback to run at different epochs. This feature is
        # useful while fine-tuning models since you can't always use the same batch size after
        # unfreezing the backbone.
        from pytorch_lightning.callbacks import BatchSizeFinder


        class FineTuneBatchSizeFinder(BatchSizeFinder):
            def __init__(self, milestones, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.milestones = milestones

            def on_fit_start(self, *args, **kwargs):
                return

            def on_train_epoch_start(self, trainer, pl_module):
                if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                    self.scale_batch_size(trainer, pl_module)


        trainer = Trainer(callbacks=[FineTuneBatchSizeFinder(milestones=(5, 10))])
        trainer.fit(...)

    Example::

        # 2. Run batch size finder for validate/test/predict.
        from pytorch_lightning.callbacks import BatchSizeFinder


        class EvalBatchSizeFinder(BatchSizeFinder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def on_fit_start(self, *args, **kwargs):
                return

            def on_test_start(self, trainer, pl_module):
                self.scale_batch_size(trainer, pl_module)


        trainer = Trainer(callbacks=[EvalBatchSizeFinder()])
        trainer.test(...)
    """

    SUPPORTED_MODES = ("power", "binsearch")

    def __init__(
        self,
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = "batch_size",
    ) -> None:
        mode = mode.lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"`mode` should be either of {self.SUPPORTED_MODES}")

        self.optimal_batch_size: Optional[int] = init_val
        self._mode = mode
        self._steps_per_trial = steps_per_trial
        self._init_val = init_val
        self._max_trials = max_trials
        self._batch_arg_name = batch_arg_name
        self._early_exit = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if trainer._accelerator_connector.is_distributed:
            raise MisconfigurationException("The Batch size finder is not supported with distributed strategies.")

        running_stage = trainer.state.stage
        assert running_stage is not None
        dl_source = getattr(trainer._data_connector, f"_{running_stage.dataloader_prefix}_dataloader_source")

        # TODO: check if this can be enabled (#4040)
        if not trainer._data_connector._train_dataloader_source.is_module():
            raise MisconfigurationException(
                "The Batch size finder cannot be used with dataloaders passed directly to `.fit()`. Please disable"
                " the feature or incorporate the dataloader into your LightningModule or LightningDataModule."
            )

        # TODO: Add support for multiple eval dataloader
        if stage != "fit":
            dataloaders = dl_source.dataloader()
            if isinstance(dataloaders, list) and len(dataloaders) > 1:
                raise MisconfigurationException(
                    f"The Batch size finder cannot be used with multiple {running_stage.dataloader_prefix} dataloaders."
                )

        if not lightning_hasattr(pl_module, self._batch_arg_name):
            raise MisconfigurationException(
                f"Field {self._batch_arg_name} not found in `model`, `datamodule`, nor their `hparams` attributes."
            )

        if (
            hasattr(pl_module, self._batch_arg_name)
            and hasattr(pl_module, "hparams")
            and self._batch_arg_name in pl_module.hparams
        ):
            rank_zero_warn(
                f"Field `model.{self._batch_arg_name}` and `model.hparams.{self._batch_arg_name}` are mutually"
                f" exclusive! `model.{self._batch_arg_name}` will be used as the initial batch size for scaling."
                " If this is not the intended behavior, please remove either one."
            )

    def scale_batch_size(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        new_size = scale_batch_size(
            trainer,
            pl_module,
            self._mode,
            self._steps_per_trial,
            self._init_val,
            self._max_trials,
            self._batch_arg_name,
        )

        self.optimal_batch_size = new_size
        if self._early_exit:
            raise _TunerExitException()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking or trainer.state.fn != "validate":
            return

        self.scale_batch_size(trainer, pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)

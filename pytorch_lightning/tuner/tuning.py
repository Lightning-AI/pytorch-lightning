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

from typing import Optional, List, Union

from torch.utils.data import DataLoader

from pytorch_lightning.tuner.batch_size_scaling import scale_batch_size
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus
from pytorch_lightning.tuner.lr_finder import _run_lr_finder_internally, lr_find
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule


class Tuner:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, auto_lr_find, auto_scale_batch_size):
        self.trainer.auto_lr_find = auto_lr_find
        self.trainer.auto_scale_batch_size = auto_scale_batch_size

    def tune(self, model, train_dataloader, val_dataloaders, datamodule):
        # setup data, etc...
        self.trainer.train_loop.setup_fit(model, train_dataloader, val_dataloaders, datamodule)

        # hook
        self.trainer.data_connector.prepare_data(model)

        # Run auto batch size scaling
        if self.trainer.auto_scale_batch_size:
            if isinstance(self.trainer.auto_scale_batch_size, bool):
                self.trainer.auto_scale_batch_size = 'power'
            self.scale_batch_size(
                model,
                mode=self.trainer.auto_scale_batch_size,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
            )
            model.logger = self.trainer.logger  # reset logger binding

        # Run learning rate finder:
        if self.trainer.auto_lr_find:
            self.internal_find_lr(model)
            model.logger = self.trainer.logger  # reset logger binding

    def scale_batch_size(
            self,
            model,
            mode: str = 'power',
            steps_per_trial: int = 3,
            init_val: int = 2,
            max_trials: int = 25,
            batch_arg_name: str = 'batch_size',
            **fit_kwargs
    ):
        r"""
        Will iteratively try to find the largest batch size for a given model
        that does not give an out of memory (OOM) error.

        Args:
            model: Model to fit.

            mode: string setting the search mode. Either `power` or `binsearch`.
                If mode is `power` we keep multiplying the batch size by 2, until
                we get an OOM error. If mode is 'binsearch', we will initially
                also keep multiplying by 2 and after encountering an OOM error
                do a binary search between the last successful batch size and the
                batch size that failed.

            steps_per_trial: number of steps to run with a given batch size.
                Idealy 1 should be enough to test if a OOM error occurs,
                however in practise a few are needed

            init_val: initial batch size to start the search with

            max_trials: max number of increase in batch size done before
               algorithm is terminated

            batch_arg_name: name of the attribute that stores the batch size.
                It is expected that the user has provided a model or datamodule that has a hyperparameter
                with that name. We will look for this attribute name in the following places

                - `model`
                - `model.hparams`
                - `model.datamodule`
                - `trainer.datamodule` (the datamodule passed to the tune method)

            **fit_kwargs: remaining arguments to be passed to .fit(), e.g., dataloader
                or datamodule.

        """
        return scale_batch_size(
            self.trainer,
            model,
            mode,
            steps_per_trial,
            init_val,
            max_trials,
            batch_arg_name,
            **fit_kwargs,
        )

    def lr_find(
            self,
            model: LightningModule,
            train_dataloader: Optional[DataLoader] = None,
            val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
            min_lr: float = 1e-8,
            max_lr: float = 1,
            num_training: int = 100,
            mode: str = 'exponential',
            early_stop_threshold: float = 4.0,
            datamodule: Optional[LightningDataModule] = None
    ):
        return lr_find(
            self.trainer,
            model,
            train_dataloader,
            val_dataloaders,
            min_lr,
            max_lr,
            num_training,
            mode,
            early_stop_threshold,
            datamodule,
        )

    def internal_find_lr(self, model: LightningModule):
        return _run_lr_finder_internally(self.trainer, model)

    def pick_multiple_gpus(self, num_gpus: int):
        return pick_multiple_gpus(num_gpus)

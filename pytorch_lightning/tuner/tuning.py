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
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.tuner.batch_size_scaling import scale_batch_size
from pytorch_lightning.tuner.lr_finder import _LRFinder, lr_find
from pytorch_lightning.utilities import rank_zero_deprecation


class Tuner:

    def __init__(self, trainer: 'pl.Trainer') -> None:
        self.trainer = trainer

    def on_trainer_init(self, auto_lr_find: Union[str, bool], auto_scale_batch_size: Union[str, bool]) -> None:
        self.trainer.auto_lr_find = auto_lr_find
        self.trainer.auto_scale_batch_size = auto_scale_batch_size

    def tune(
        self,
        model: 'pl.LightningModule',
        scale_batch_size_kwargs: Optional[Dict[str, Any]] = None,
        lr_find_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Union[int, _LRFinder]]:
        scale_batch_size_kwargs = scale_batch_size_kwargs or {}
        lr_find_kwargs = lr_find_kwargs or {}
        result = None

        # Run auto batch size scaling
        if self.trainer.auto_scale_batch_size:
            result = scale_batch_size(self.trainer, model, **scale_batch_size_kwargs)

        # Run learning rate finder:
        if self.trainer.auto_lr_find:
            lr_find_kwargs.setdefault('update_attr', True)
            lr_find(self.trainer, model, **lr_find_kwargs)

        self.trainer.state = TrainerState.FINISHED

        return result

    def _launch(self, *args: Any, **kwargs: Any) -> None:
        """`_launch` wrapper to set the proper state during tuning, as this can be called multiple times"""
        self.trainer.state = TrainerState.TUNING  # last `_launch` call might have set it to `FINISHED`
        self.trainer.training = True
        self.trainer._launch(*args, **kwargs)
        self.trainer.tuning = True

    def scale_batch_size(
        self,
        model: 'pl.LightningModule',
        mode: str = 'power',
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = 'batch_size',
        **fit_kwargs
    ):
        rank_zero_deprecation(
            "`Tuner.scale_batch_size()` is deprecated in v1.3 and will be removed in v1.5."
            " Please use `trainer.tune(scale_batch_size_kwargs={...})` instead."
        )
        self.trainer.auto_lr_find = True
        return self.trainer.tune(
            model,
            **fit_kwargs,
            scale_batch_size_kwargs={
                'mode': mode,
                'steps_per_trial': steps_per_trial,
                'init_val': init_val,
                'max_trials': max_trials,
                'batch_arg_name': batch_arg_name,
            }
        )

    def lr_find(
        self,
        model: 'pl.LightningModule',
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training: int = 100,
        mode: str = 'exponential',
        early_stop_threshold: float = 4.0,
        datamodule: Optional['pl.LightningDataModule'] = None,
        update_attr: bool = False,
    ):
        rank_zero_deprecation(
            "`Tuner.lr_find()` is deprecated in v1.3 and will be removed in v1.5."
            " Please use `trainer.tune(lr_finder_kwargs={...})` instead."
        )
        self.trainer.auto_scale_batch_size = True
        return self.trainer.tune(
            model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            lr_find_kwargs={
                'min_lr': min_lr,
                'max_lr': max_lr,
                'num_training': num_training,
                'mode': mode,
                'early_stop_threshold': early_stop_threshold,
                'update_attr': update_attr
            }
        )

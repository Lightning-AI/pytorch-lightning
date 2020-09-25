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
from pytorch_lightning.tuner.batch_size_scaling import scale_batch_size
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus
from pytorch_lightning.tuner.lr_finder import _run_lr_finder_internally, lr_find
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from typing import Optional, List, Union
from torch.utils.data import DataLoader


class Tuner:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, auto_lr_find, auto_scale_batch_size):
        self.trainer.auto_lr_find = auto_lr_find
        self.trainer.auto_scale_batch_size = auto_scale_batch_size

    def scale_batch_size(self,
                         model,
                         mode: str = 'power',
                         steps_per_trial: int = 3,
                         init_val: int = 2,
                         max_trials: int = 25,
                         batch_arg_name: str = 'batch_size',
                         **fit_kwargs):
        return scale_batch_size(
            self.trainer, model, mode, steps_per_trial, init_val, max_trials, batch_arg_name, **fit_kwargs
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

    def internal_find_lr(self, trainer, model: LightningModule):
        return _run_lr_finder_internally(trainer, model)

    def pick_multiple_gpus(self, num_gpus: int):
        return pick_multiple_gpus(num_gpus)

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

from typing import Callable, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import prefetch_iterator
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class DataConnector:
    def __init__(self, trainer: "pl.Trainer", multiple_trainloader_mode: str = "max_size_cycle"):
        self.trainer = trainer
        self.multiple_trainloader_mode = multiple_trainloader_mode

    def on_trainer_init(
        self,
        check_val_every_n_epoch: int,
        reload_dataloaders_every_n_epochs: int,
        reload_dataloaders_every_epoch: bool,
        prepare_data_per_node: bool,
    ) -> None:
        self.trainer.datamodule = None
        self.trainer.prepare_data_per_node = prepare_data_per_node

        if not isinstance(check_val_every_n_epoch, int):
            raise MisconfigurationException(
                f"check_val_every_n_epoch should be an integer. Found {check_val_every_n_epoch}"
            )

        self.trainer.check_val_every_n_epoch = check_val_every_n_epoch

        if reload_dataloaders_every_epoch:
            reload_dataloaders_every_n_epochs = int(reload_dataloaders_every_epoch)
            rank_zero_deprecation(
                "`reload_dataloaders_every_epoch` is deprecated in v1.4 and will be removed in v1.6."
                " Please use `reload_dataloaders_every_n_epochs` in Trainer."
            )

        if not isinstance(reload_dataloaders_every_n_epochs, int) or (reload_dataloaders_every_n_epochs < 0):
            raise MisconfigurationException(
                "`reload_dataloaders_every_n_epochs` should be an int >= 0,"
                f" got {reload_dataloaders_every_n_epochs}."
            )

        self.trainer.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        self.trainer._is_data_prepared = False

    def get_profiled_train_dataloader(self, train_dataloader):
        profiled_dl = self.trainer.profiler.profile_iterable(
            enumerate(prefetch_iterator(train_dataloader)), "get_train_batch"
        )
        return profiled_dl

    def prepare_data(self, model):
        # on multi-gpu jobs we only want to manipulate (download, etc) on node_rank=0, local_rank=0
        # or in the case where each node needs to do its own manipulation in which case just local_rank=0
        if self.can_prepare_data():
            if self.trainer.datamodule is not None:
                self.trainer.datamodule.prepare_data()
            model.prepare_data()
            self.trainer._is_data_prepared = True

    def can_prepare_data(self):
        should_call_dm_prepare_data = True
        if self.trainer.datamodule is not None and is_overridden("prepare_data", self.trainer.datamodule):
            should_call_dm_prepare_data = not self.trainer.datamodule._has_prepared_data

        if self.trainer.prepare_data_per_node:
            return self.trainer.local_rank == 0 and should_call_dm_prepare_data
        return self.trainer.node_rank == 0 and self.trainer.local_rank == 0 and should_call_dm_prepare_data

    def attach_data(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[TRAIN_DATALOADERS] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        test_dataloaders: Optional[EVAL_DATALOADERS] = None,
        predict_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional["pl.LightningDataModule"] = None,
    ) -> None:
        # set up the passed in dataloaders (if needed)
        self.attach_dataloaders(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            test_dataloaders=test_dataloaders,
            predict_dataloaders=predict_dataloaders,
        )
        self.attach_datamodule(model, datamodule=datamodule)
        # set local properties on the model
        self.trainer.model_connector.copy_trainer_model_properties(model)

    def attach_dataloaders(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[TRAIN_DATALOADERS] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        test_dataloaders: Optional[EVAL_DATALOADERS] = None,
        predict_dataloaders: Optional[EVAL_DATALOADERS] = None,
    ) -> None:
        # when dataloader is passed via fit, patch the train_dataloader
        # functions to overwrite with these implementations
        if train_dataloaders is not None:
            self.trainer.train_dataloader = None
            train_dataloader = _PatchDataLoader(train_dataloaders, "train")
            train_dataloader.patch(model)

        if val_dataloaders is not None:
            self.trainer.val_dataloaders = None
            val_dataloader = _PatchDataLoader(val_dataloaders, "val")
            val_dataloader.patch(model)

        if test_dataloaders is not None:
            self.trainer.test_dataloaders = None
            test_dataloader = _PatchDataLoader(test_dataloaders, "test")
            test_dataloader.patch(model)

        if predict_dataloaders is not None:
            self.trainer.predict_dataloaders = None
            predict_dataloader = _PatchDataLoader(predict_dataloaders, "predict")
            predict_dataloader.patch(model)

    def attach_datamodule(
        self, model: "pl.LightningModule", datamodule: Optional["pl.LightningDataModule"] = None
    ) -> None:
        # If we have a datamodule, attach necessary hooks + dataloaders
        if datamodule is None:
            return

        # Override loader hooks
        dl_methods = ("train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader")
        for method in dl_methods:
            if is_overridden(method, datamodule):
                setattr(model, method, getattr(datamodule, method))

        # Override data transfer hooks if dataset-specific to_device logic has been defined in datamodule
        batch_transfer_hooks = ("on_before_batch_transfer", "transfer_batch_to_device", "on_after_batch_transfer")
        for hook in batch_transfer_hooks:
            if is_overridden(hook, datamodule):
                setattr(model, hook, getattr(datamodule, hook))

        self.trainer.datamodule = datamodule
        datamodule.trainer = self.trainer

        # experimental feature for Flash
        if hasattr(datamodule, "data_pipeline"):
            model.data_pipeline = datamodule.data_pipeline

    @staticmethod
    def detach_data(model: "pl.LightningModule") -> None:
        for stage in ("train", "val", "test", "predict"):
            loader = getattr(model, f"{stage}_dataloader", None)
            if isinstance(loader, _PatchDataLoader):
                loader.unpatch(model)


class _PatchDataLoader:
    r"""
    Callable object for patching dataloaders passed into trainer.fit().
    Use this class to override model.*_dataloader() and be pickle-compatible.

    Args:
        dataloader: Dataloader object to return when called.
    """

    def __init__(self, dataloader: Union[TRAIN_DATALOADERS, EVAL_DATALOADERS], stage: str) -> None:
        self.dataloader = dataloader

        # cannot pickle __code__ so cannot verify if PatchDataloader
        # exists which shows dataloader methods have been overwritten.
        # so, we hack it by using the string representation
        self.patch_loader_code = str(self.__call__.__code__)
        self.old_loader: Optional[Callable] = None
        self.stage = stage

    def __call__(self) -> Union[TRAIN_DATALOADERS, EVAL_DATALOADERS]:
        return self.dataloader

    def patch(self, model: "pl.LightningModule") -> None:
        self._old_loader = getattr(model, self.stage + "_dataloader")
        setattr(model, self.stage + "_dataloader", self)

    def unpatch(self, model: "pl.LightningModule") -> None:
        setattr(model, self.stage + "_dataloader", self._old_loader)
        self._old_loader = None

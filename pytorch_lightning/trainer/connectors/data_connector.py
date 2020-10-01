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

from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from typing import List, Optional, Union
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.model_utils import is_overridden


class DataConnector(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, check_val_every_n_epoch, reload_dataloaders_every_epoch, prepare_data_per_node):
        self.trainer.datamodule = None
        self.trainer.prepare_data_per_node = prepare_data_per_node

        self.trainer.check_val_every_n_epoch = check_val_every_n_epoch
        self.trainer.reload_dataloaders_every_epoch = reload_dataloaders_every_epoch
        self.trainer._is_data_prepared = False

    def get_profiled_train_dataloader(self, train_dataloader):
        profiled_dl = self.trainer.profiler.profile_iterable(
            enumerate(self._with_is_last(train_dataloader)),
            "get_train_batch"
        )
        return profiled_dl

    def _with_is_last(self, iterable):
        """Pass through values from the given iterable with an added boolean indicating if this is the last item.
        See `https://stackoverflow.com/a/1630350 <https://stackoverflow.com/a/1630350>`_"""
        it = iter(iterable)
        last = next(it)
        for val in it:
            # yield last and has next
            yield last, False
            last = val
        # yield last, no longer has next
        yield last, True

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
        if self.trainer.datamodule is not None and is_overridden('prepare_data', self.trainer.datamodule):
            should_call_dm_prepare_data = not self.trainer.datamodule.has_prepared_data

        if self.trainer.prepare_data_per_node:
            return self.trainer.local_rank == 0 and should_call_dm_prepare_data
        else:
            return self.trainer.node_rank == 0 and self.trainer.local_rank == 0 and should_call_dm_prepare_data

    def attach_data(self, model, train_dataloader, val_dataloaders, datamodule):
        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloader, LightningDataModule):
            datamodule = train_dataloader
            train_dataloader = None

        self.__enforce_datamodule_dataloader_override(train_dataloader, val_dataloaders, datamodule)

        # set up the passed in dataloaders (if needed)
        self.attach_dataloaders(model, train_dataloader, val_dataloaders)
        self.attach_datamodule(model, datamodule, 'fit')

    def __enforce_datamodule_dataloader_override(self, train_dataloader, val_dataloaders, datamodule):
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloader is not None or val_dataloaders is not None) and datamodule is not None:
            raise MisconfigurationException(
                'You cannot pass train_dataloader or val_dataloaders to trainer.fit if you supply a datamodule'
            )

    def attach_dataloaders(self, model, train_dataloader=None, val_dataloaders=None, test_dataloaders=None):
        # when dataloader is passed via fit, patch the train_dataloader
        # functions to overwrite with these implementations
        if train_dataloader is not None:
            model.train_dataloader = _PatchDataLoader(train_dataloader)

        if val_dataloaders is not None:
            model.val_dataloader = _PatchDataLoader(val_dataloaders)

        if test_dataloaders is not None:
            model.test_dataloader = _PatchDataLoader(test_dataloaders)

    def attach_datamodule(self, model, datamodule: Optional[LightningDataModule], stage: str) -> None:

        # We use datamodule if it's been provided on .fit or .test, otherwise we check model for it
        datamodule = datamodule or getattr(model, 'datamodule', None)

        # If we have a datamodule, attach necessary hooks + dataloaders
        if datamodule:

            # Override loader hooks
            if is_overridden('train_dataloader', datamodule):
                model.train_dataloader = datamodule.train_dataloader
            if is_overridden('val_dataloader', datamodule):
                model.val_dataloader = datamodule.val_dataloader
            if is_overridden('test_dataloader', datamodule):
                model.test_dataloader = datamodule.test_dataloader

            # Override transfer_batch_to_device if dataset-specific to_device logic has been defined in datamodule
            if is_overridden('transfer_batch_to_device', datamodule):
                model.transfer_batch_to_device = datamodule.transfer_batch_to_device

            self.trainer.datamodule = datamodule
            datamodule.trainer = self.trainer


class _PatchDataLoader(object):
    r"""
    Callable object for patching dataloaders passed into trainer.fit().
    Use this class to override model.*_dataloader() and be pickle-compatible.

    Args:
        dataloader: Dataloader object to return when called.

    """

    def __init__(self, dataloader: Union[List[DataLoader], DataLoader]):
        self.dataloader = dataloader

        # cannot pickle __code__ so cannot verify if PatchDataloader
        # exists which shows dataloader methods have been overwritten.
        # so, we hack it by using the string representation
        self.patch_loader_code = str(self.__call__.__code__)

    def __call__(self) -> Union[List[DataLoader], DataLoader]:
        return self.dataloader

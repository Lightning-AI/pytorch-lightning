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
from typing import List, Union
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.model_utils import is_overridden


class DataConnector(object):

    def __init__(self, trainer):
        self.trainer = trainer

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

    def attach_datamodule(self, model, datamodule, stage):

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

            self.datamodule = datamodule


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

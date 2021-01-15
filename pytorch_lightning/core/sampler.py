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

from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_warn
import inspect


class LightningBatchSamplerWrapper:
    """
    This class wraps user batch sampler, so we can extract 
    the batch_indices for tracking each sample.
    """

    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler
        self.batch_indices = None

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            self.batch_indices = batch_indices
            print(batch_indices)
            yield batch_indices

    @staticmethod
    def to_new_dataloader(dataloader: DataLoader) -> DataLoader:
        """
        This function will wrap the user batch_sampler to track the returned batch indices
        """
        if not isinstance(dataloader, DataLoader):
            raise MisconfigurationException('Autoid only works with torch dataloaders or derived classes!')
        
        if isinstance(dataloader.dataset, IterableDataset):
            return dataloader

        skip_keys = ['sampler', 'batch_sampler', 'dataset_kind']
        skip_valid_keys = ['args', 'kwargs', 'self']

        params = {k:v for k, v in vars(dataloader).items() if not k.startswith("_")}
        
        valid_kwargs = [*inspect.signature(dataloader.__init__).parameters]
        if isinstance(dataloader, DataLoader):
            valid_kwargs += [*inspect.signature(DataLoader.__init__).parameters]
        valid_kwargs = inspect.signature(dataloader.__init__).parameters
        valid_kwargs = set(valid_kwargs)
        
        dl_args = dict(
            (name, params[name]) for name in valid_kwargs 
            if name in params and name not in skip_keys
        )

        multiprocessing_context = dataloader.multiprocessing_context
        
        # override parameters to enable batch_sampler injection
        dl_args["batch_size"] = 1
        dl_args["sampler"] = None
        dl_args["shuffle"] = None
        dl_args["batch_sampler"] = LightningBatchSamplerWrapper(dataloader.batch_sampler)
        dl_args["drop_last"] = False
        dl_args['multiprocessing_context'] = multiprocessing_context

        missing_kwargs = valid_kwargs.difference(skip_valid_keys).difference(set(dl_args))
        if len(missing_kwargs) != 0:
            dataloader_cls_name = dataloader.__class__.__name__
            rank_zero_warn(
                f"Trying to replace your BatchSampler for {dataloader_cls_name} dataloader."
                "This would fail as your DataLoader doesn't expose as attributes all its __init__ parameters. "
                f"Missing attributes are {missing_kwargs}", UserWarning
            )
            return dataloader
        dataloader = type(dataloader)(**dl_args)
        dataloader.multiprocessing_context = multiprocessing_context
        return dataloader

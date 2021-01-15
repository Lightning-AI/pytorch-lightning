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

import inspect

from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader


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
    def to_new_dataloader(dataloader) -> DataLoader:
        """
        This function will wrap the user batch_sampler to track the returned batch indices
        """
        dataset = dataloader.dataset
        if isinstance(dataset, IterableDataset):
            return dataloader

        if getattr(dataloader, "dataset", None) is not None and getattr(dataloader, "batch_sampler", None) is not None:
            dl_args = {
                "batch_size": 1,
                "sampler": None,
                "batch_sampler": LightningBatchSamplerWrapper(dataloader.batch_sampler),
                "num_workers": getattr(dataloader, "num_workers", 0),
                "collate_fn": getattr(dataloader, "collate_fn", None),
                "pin_memory": getattr(dataloader, "pin_memory", False),
                "drop_last": False,
                "timeout": getattr(dataloader, "timeout", 0),
                "worker_init_fn": getattr(dataloader, "worker_init_fn", None),
                "multiprocessing_context": getattr(dataloader, "multiprocessing_context", None),
            }

            dataset = dataloader.dataset

            params = vars(dataloader)

            valid_kwargs = inspect.signature(dataloader.__init__).parameters
            extra_args = dict(
                (name, params[name]) for name in valid_kwargs
                if name in params and name not in dl_args and name != 'dataset'
            )
            dl_args.update(**extra_args)

            dataloader = type(dataloader)(dataset, **dl_args)
        return dataloader

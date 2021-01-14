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
from torch.utils.data.sampler import BatchSampler


class LightningBatchSampler(BatchSampler):
    """
    This sampler is used to capture indices from the sampler.
    """

    batch_indices = None

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                self.batch_indices = batch
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            self.batch_indices = batch
            yield batch

    @staticmethod
    def to_new_dataloader(dataloader) -> DataLoader:
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None or isinstance(dataset, IterableDataset):
            return dataloader

        if getattr(dataloader, "sampler", None) is not None:
            dl_args = {
                "batch_size": 1,
                "sampler": None,
                "batch_sampler": LightningBatchSampler(dataloader.sampler, dataloader.batch_size, dataloader.drop_last),
                "num_workers": getattr(dataloader, "num_workers", 0),
                "collate_fn": getattr(dataloader, "collate_fn", None),
                "pin_memory": getattr(dataloader, "pin_memory", False),
                "drop_last": getattr(dataloader, "drop_last", False),
                "timeout": getattr(dataloader, "timeout", 0),
                "worker_init_fn": getattr(dataloader, "worker_init_fn", None),
                "multiprocessing_context": getattr(dataloader, "multiprocessing_context", None),
            }

            dataset = dataloader.dataset
            dataloader = type(dataloader)(dataset, **dl_args)
        return dataloader

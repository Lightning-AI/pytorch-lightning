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

    @classmethod
    def to_new_dataloader(cls, dataloader, shuffle=False):
        return type(dataloader)(
            dataloader.dataset, 
            batch_size=1, 
            shuffle=None, 
            sampler=None,
            batch_sampler=cls(dataloader.sampler, dataloader.batch_size, dataloader.drop_last), 
            num_workers=dataloader.num_workers, 
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory, 
            drop_last=None, 
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn, 
            multiprocessing_context=dataloader.multiprocessing_context,
            generator=dataloader.generator            
        )
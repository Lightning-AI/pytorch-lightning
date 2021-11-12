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
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import BatchSampler, Sampler, SequentialSampler

from pl_examples.basic_examples.mnist_datamodule import MNIST
from pl_examples.basic_examples.mnist_examples.image_classifier_4_lightning_module import ImageClassifier
from pytorch_lightning import seed_everything, Trainer
from tests.helpers.datasets import make_unbalanced

train_dataset = MNIST(root="./data", train=True, download=True, transform=T.ToTensor())
test_dataset = MNIST(root="./data", train=False, download=True, transform=T.ToTensor())

train_dataset_unbalanced = make_unbalanced(
    train_dataset, weights=[0.29, 0.01, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05, 0.15, 0.15]
)


class WeightedRandomBatchSampler(Sampler):
    def __init__(self, dataset: Subset, batch_size: int, drop_last: bool, seed: int = 42, replacement: bool = False):
        self.num_samples = len(dataset)
        bicount = torch.tensor(np.bincount(dataset.targets)).float()
        bicount /= bicount.sum()
        self.weights = torch.zeros((self.num_samples,))
        for idx, count in enumerate(bicount):
            mask = np.where(dataset.targets == idx)[0]
            self.weights[mask] = count / mask.sum()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.replacement = replacement
        self.generator = torch.Generator()
        self.iteration_counter = 0
        self.restart_counter = 0
        self.cache_state_dict = None

    def state_dict(self) -> Dict:
        return {"random_state": self.generator.get_state(), "iteration_counter": self.iteration_counter}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.cache_state_dict = state_dict

    def __iter__(self) -> "WeightedRandomBatchSampler":
        self.restart_counter = 0
        if self.cache_state_dict:
            self.generator.set_state(state_dict["random_state"])
            self.iteration_counter = state_dict["iteration_counter"]
            self.cache_state_dict = None
        else:
            self.iteration_counter = 0
        indices = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        self.sampler_iter = iter(
            BatchSampler(SequentialSampler(indices), batch_size=self.batch_size, drop_last=self.drop_last)
        )
        return self

    def __next__(self):
        while self.restart_counter < self.iteration_counter:
            next(self.sampler_iter)
            self.restart_counter += 1

        batch = next(self.sampler_iter)
        self.iteration_counter += 1
        self.restart_counter += 1
        return batch


seed_everything(42)

batch_sampler = WeightedRandomBatchSampler(train_dataset_unbalanced, 2, False)
train_dataloader = DataLoader(train_dataset_unbalanced, batch_sampler=batch_sampler)

# VALIDATE OUR IMPLEMENTATION IS PROPERLY WORKING

for idx, batch in enumerate(train_dataloader):
    if idx == 100:
        state_dict = batch_sampler.state_dict()

batch_sampler = WeightedRandomBatchSampler(train_dataset_unbalanced, 2, False)
batch_sampler.load_state_dict(state_dict)
train_dataloader = DataLoader(train_dataset_unbalanced, batch_sampler=batch_sampler)

for idx, reloaded_batch in enumerate(train_dataloader):
    pass

assert torch.equal(batch[0], reloaded_batch[0])
assert torch.equal(batch[1], reloaded_batch[1])

model = ImageClassifier()
trainer = Trainer(max_epochs=2)
trainer.fit(model, train_dataloader)

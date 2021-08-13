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
import pytest
from torch import tensor
from torch.utils.data import DataLoader, IterableDataset

from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.fetching import LightningFetcher


@pytest.mark.parametrize("use_combined_loader", [False, True])
def test_prefetch_iterator(use_combined_loader):
    """Test the LightningFetcher with PyTorch IterableDataset."""

    class IterDataset(IterableDataset):
        def __iter__(self):
            yield 1
            yield 2
            yield 3

    for prefetch_batches in range(1, 5):
        if use_combined_loader:
            loader = CombinedLoader([DataLoader(IterDataset()), DataLoader(IterDataset())])
            expected = [
                ([tensor([1]), tensor([1])], False),
                ([tensor([2]), tensor([2])], False),
                ([tensor([3]), tensor([3])], True),
            ]
        else:
            loader = DataLoader(IterDataset())
            expected = [(1, False), (2, False), (3, True)]
        iterator = LightningFetcher(prefetch_batches=prefetch_batches)
        iterator.setup(loader)

        def generate():
            generated = []
            for idx, data in enumerate(iterator, 1):
                if iterator.done:
                    assert iterator.fetched == 3
                else:
                    assert iterator.fetched == (idx + prefetch_batches)
                generated.append(data)
            return generated

        assert generate() == expected
        # validate reset works properly.
        assert generate() == expected
        assert iterator.fetched == 3

    class EmptyIterDataset(IterableDataset):
        def __iter__(self):
            return iter([])

    dataloader = DataLoader(EmptyIterDataset())
    iterator = LightningFetcher()
    iterator.setup(dataloader)
    assert list(iterator) == []

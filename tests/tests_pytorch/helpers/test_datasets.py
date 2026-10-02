# Copyright The Lightning AI team.
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
import pickle

import cloudpickle
import pytest
import torch

from tests_pytorch.helpers.datasets import MNIST, AverageDataset, TrialMNIST


@pytest.mark.parametrize("train", [False, True])
def test_mnist(tmp_path, mock_mnist_download, train):
    dataset = MNIST(tmp_path, train=train, download=True)
    images, targets = mock_mnist_download[dataset.TRAIN_FILE_NAME if train else dataset.TEST_FILE_NAME]
    assert len(dataset) == len(images)
    torch.testing.assert_close(dataset.data, images)
    torch.testing.assert_close(dataset.targets, targets)
    image, target = dataset[0]
    torch.testing.assert_close(image, (images[0].float().unsqueeze(0) - 0.1307) / 0.3081)
    assert target == targets[0].item()


@pytest.mark.parametrize(("train", "num_samples"), [(True, 100), (False, 10)])
def test_trial_mnist(tmp_path, mock_mnist_download, train, num_samples):
    dataset = TrialMNIST(tmp_path, train=train, num_samples=num_samples, download=True)
    assert len(dataset) == 3 * num_samples
    assert set(dataset.targets.tolist()) == {0, 1, 2}
    assert torch.bincount(dataset.targets).tolist() == [num_samples] * 3
    images, targets = mock_mnist_download[dataset.TRAIN_FILE_NAME if train else dataset.TEST_FILE_NAME]
    indices = torch.cat([(targets == digit).nonzero().flatten()[:num_samples] for digit in (0, 1, 2)]).sort().values
    torch.testing.assert_close(dataset.data, images[indices])
    torch.testing.assert_close(dataset.targets, targets[indices])


@pytest.mark.parametrize("dataset_cls", [MNIST, TrialMNIST, AverageDataset])
@pytest.mark.parametrize("pickle_module", [pickle, cloudpickle])
def test_pickling_dataset_mnist(tmp_path, mock_mnist_download, dataset_cls, pickle_module):
    args = {"root": tmp_path} if issubclass(dataset_cls, MNIST) else {}
    mnist = dataset_cls(**args)

    restored = pickle_module.loads(pickle_module.dumps(mnist))
    assert len(restored) == len(mnist)
    torch.testing.assert_close(restored[0], mnist[0])

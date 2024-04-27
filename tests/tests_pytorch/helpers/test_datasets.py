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

from tests_pytorch import _PATH_DATASETS
from tests_pytorch.helpers.datasets import MNIST, AverageDataset, TrialMNIST


def test_mnist(tmp_path):
    dataset = MNIST(tmp_path, download=True)
    assert len(dataset) == 60000
    assert torch.bincount(dataset.targets).tolist() == [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]


def test_trial_mnist(tmp_path):
    dataset = TrialMNIST(tmp_path, download=True)
    assert len(dataset) == 300
    assert set(dataset.targets.tolist()) == {0, 1, 2}
    assert torch.bincount(dataset.targets).tolist() == [100, 100, 100]


@pytest.mark.parametrize(
    ("dataset_cls", "args"),
    [(MNIST, {"root": _PATH_DATASETS}), (TrialMNIST, {"root": _PATH_DATASETS}), (AverageDataset, {})],
)
def test_pickling_dataset_mnist(dataset_cls, args):
    mnist = dataset_cls(**args)

    mnist_pickled = pickle.dumps(mnist)
    pickle.loads(mnist_pickled)
    # assert vars(mnist) == vars(mnist_loaded)

    mnist_pickled = cloudpickle.dumps(mnist)
    cloudpickle.loads(mnist_pickled)
    # assert vars(mnist) == vars(mnist_loaded)

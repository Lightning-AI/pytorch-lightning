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
import pickle

import cloudpickle
import pytest

from tests.base.datasets import MNIST, TrialMNIST, AverageDataset


@pytest.mark.parametrize('dataset_cls', [MNIST, TrialMNIST, AverageDataset])
def test_pickling_dataset_mnist(tmpdir, dataset_cls):
    mnist = dataset_cls()

    mnist_pickled = pickle.dumps(mnist)
    mnist_loaded = pickle.loads(mnist_pickled)
    # assert vars(mnist) == vars(mnist_loaded)

    mnist_pickled = cloudpickle.dumps(mnist)
    mnist_loaded = cloudpickle.loads(mnist_pickled)
    # assert vars(mnist) == vars(mnist_loaded)

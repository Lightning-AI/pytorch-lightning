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

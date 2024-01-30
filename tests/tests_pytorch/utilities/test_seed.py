import random
from unittest import mock

import numpy as np
import pytest
import torch
from lightning.pytorch.utilities.seed import isolate_rng

from tests_pytorch.helpers.runif import RunIf


@pytest.mark.parametrize("with_torch_cuda", [False, pytest.param(True, marks=RunIf(min_cuda_gpus=1))])
def test_isolate_rng(with_torch_cuda):
    """Test that the isolate_rng context manager isolates the random state from the outer scope."""
    # torch
    torch.rand(1)
    with isolate_rng():
        generated = [torch.rand(2) for _ in range(3)]
    assert torch.equal(torch.rand(2), generated[0])

    # torch.cuda
    if with_torch_cuda:
        torch.cuda.FloatTensor(1).normal_()
        with isolate_rng():
            generated = [torch.cuda.FloatTensor(2).normal_() for _ in range(3)]
        assert torch.equal(torch.cuda.FloatTensor(2).normal_(), generated[0])

    # numpy
    np.random.rand(1)
    with isolate_rng():
        generated = [np.random.rand(2) for _ in range(3)]
    assert np.equal(np.random.rand(2), generated[0]).all()

    # python
    random.random()
    with isolate_rng():
        generated = [random.random() for _ in range(3)]
    assert random.random() == generated[0]


@mock.patch("torch.cuda.set_rng_state_all")
@mock.patch("torch.cuda.get_rng_state_all")
def test_isolate_rng_cuda(get_cuda_rng, set_cuda_rng):
    """Test that `include_cuda` controls whether isolate_rng also manages torch.cuda's rng."""
    with isolate_rng(include_cuda=False):
        get_cuda_rng.assert_not_called()
    set_cuda_rng.assert_not_called()

    with isolate_rng(include_cuda=True):
        assert get_cuda_rng.call_count == int(torch.cuda.is_available())
    set_cuda_rng.assert_called_once()

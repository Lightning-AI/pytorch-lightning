import random

import numpy as np
import torch

from pytorch_lightning.utilities.seed import isolate_rng


def test_isolate_rng():
    """Test that the isolate_rng context manager isolates the random state from the outer scope."""
    # torch
    torch.rand(1)
    with isolate_rng():
        generated = [torch.rand(2) for _ in range(3)]
    assert torch.equal(torch.rand(2), generated[0])

    # torch.cuda
    if torch.cuda.is_available():
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

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score as sk_average_precision

from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional.ir_average_precision import retrieval_average_precision


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    pytest.param(sk_average_precision, retrieval_average_precision),
])
def test_against_sklearn(sklearn_metric, torch_metric):
    """Compare PL metrics to sklearn version. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    rounds = 25
    sizes = [1, 4, 10, 100]

    for size in sizes:
        for _ in range(rounds):
            a = np.random.randn(size)
            b = np.random.randn(size) > 0

            sk = torch.tensor(sklearn_metric(b, a), device=device)
            pl = torch_metric(torch.tensor(a, device=device), torch.tensor(b, device=device))

            # `torch_metric`s return 0 when no label is True
            # while `sklearn.average_precision_score` returns NaN
            if math.isnan(sk):
                assert pl == 0
            else:
                assert torch.allclose(sk.float(), pl.float())

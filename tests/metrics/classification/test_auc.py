from collections import namedtuple
from functools import partial

import pytest
import torch
import numpy as np

from pytorch_lightning.metrics.classification.auc import AUC
from pytorch_lightning.metrics.functional.auc import auc
from sklearn.metrics import auc as _sk_auc
from tests.metrics.utils import MetricTester, NUM_BATCHES

torch.manual_seed(42)


def sk_auc(x, y, reorder=False):
    x = x.flatten()
    y = y.flatten()
    if reorder:
        idx = np.argsort(x, kind='stable')
        x = x[idx]
        y = y[idx]
    return _sk_auc(x, y)


Input = namedtuple('Input', ["x", "y", "reorder"])

_examples = []
# generate already ordered samples, sorted in both directions
for i in range(4):
    x = np.random.randint(0, 5, (NUM_BATCHES * 8))
    y = np.random.randint(0, 5, (NUM_BATCHES * 8))
    idx = np.argsort(x, kind='stable')
    x = x[idx] if i % 2 == 0 else x[idx[::-1]]
    y = y[idx] if i % 2 == 0 else x[idx[::-1]]
    x = x.reshape(NUM_BATCHES, 8)
    y = y.reshape(NUM_BATCHES, 8)
    _examples.append(Input(x=torch.tensor(x), y=torch.tensor(y), reorder=False))


@pytest.mark.parametrize("x, y, reorder", _examples)
class TestAUC(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_auc(self, x, y, reorder, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=x,
            target=y,
            metric_class=AUC,
            sk_metric=partial(sk_auc, reorder=reorder),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"reorder": reorder}
        )

    def test_auc_functional(self, x, y, reorder):
        self.run_functional_metric_test(
            x,
            y,
            metric_functional=auc,
            sk_metric=partial(sk_auc, reorder=reorder),
            metric_args={"reorder": reorder}
        )


@pytest.mark.parametrize(['x', 'y', 'expected'], [
    pytest.param([0, 1], [0, 1], 0.5),
    pytest.param([1, 0], [0, 1], 0.5),
    pytest.param([1, 0, 0], [0, 1, 1], 0.5),
    pytest.param([0, 1], [1, 1], 1),
    pytest.param([0, 0.5, 1], [0, 0.5, 1], 0.5),
])
def test_auc(x, y, expected):
    # Test Area Under Curve (AUC) computation
    assert auc(torch.tensor(x), torch.tensor(y)) == expected

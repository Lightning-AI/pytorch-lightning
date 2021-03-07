from collections import namedtuple
from functools import partial

import pytest
import torch
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_squared_error as sk_mean_squared_error
from sklearn.metrics import mean_squared_log_error as sk_mean_squared_log_error

from pytorch_lightning.metrics.functional import mean_absolute_error, mean_squared_error, mean_squared_log_error
from pytorch_lightning.metrics.regression import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogError
from tests.metrics.utils import BATCH_SIZE, MetricTester, NUM_BATCHES

torch.manual_seed(42)

num_targets = 5

Input = namedtuple('Input', ["preds", "target"])

_single_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)


def _single_target_sk_metric(preds, target, sk_fn=mean_squared_error):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return sk_fn(sk_preds, sk_target)


def _multi_target_sk_metric(preds, target, sk_fn=mean_squared_error):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    return sk_fn(sk_preds, sk_target)


@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric),
    ],
)
@pytest.mark.parametrize(
    "metric_class, metric_functional, sk_fn",
    [
        (MeanSquaredError, mean_squared_error, sk_mean_squared_error),
        (MeanAbsoluteError, mean_absolute_error, sk_mean_absolute_error),
        (MeanSquaredLogError, mean_squared_log_error, sk_mean_squared_log_error),
    ],
)
class TestMeanError(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_mean_error_class(
        self, preds, target, sk_metric, metric_class, metric_functional, sk_fn, ddp, dist_sync_on_step
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(sk_metric, sk_fn=sk_fn),
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_mean_error_functional(self, preds, target, sk_metric, metric_class, metric_functional, sk_fn):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            sk_metric=partial(sk_metric, sk_fn=sk_fn),
        )


@pytest.mark.parametrize("metric_class", [MeanSquaredError, MeanAbsoluteError, MeanSquaredLogError])
def test_error_on_different_shape(metric_class):
    metric = metric_class()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(100, ), torch.randn(50, ))

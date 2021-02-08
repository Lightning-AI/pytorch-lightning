from collections import namedtuple
from functools import partial

import pytest
import torch
from sklearn.metrics import explained_variance_score

from pytorch_lightning.metrics.functional import explained_variance
from pytorch_lightning.metrics.regression import ExplainedVariance
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


def _single_target_sk_metric(preds, target, sk_fn=explained_variance_score):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return sk_fn(sk_target, sk_preds)


def _multi_target_sk_metric(preds, target, sk_fn=explained_variance_score):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    return sk_fn(sk_target, sk_preds)


@pytest.mark.parametrize("multioutput", ['raw_values', 'uniform_average', 'variance_weighted'])
@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric),
    ],
)
class TestExplainedVariance(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_explained_variance(self, multioutput, preds, target, sk_metric, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            ExplainedVariance,
            partial(sk_metric, sk_fn=partial(explained_variance_score, multioutput=multioutput)),
            dist_sync_on_step,
            metric_args=dict(multioutput=multioutput),
        )

    def test_explained_variance_functional(self, multioutput, preds, target, sk_metric):
        self.run_functional_metric_test(
            preds,
            target,
            explained_variance,
            partial(sk_metric, sk_fn=partial(explained_variance_score, multioutput=multioutput)),
            metric_args=dict(multioutput=multioutput),
        )


def test_error_on_different_shape(metric_class=ExplainedVariance):
    metric = metric_class()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(100, ), torch.randn(50, ))

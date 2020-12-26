from collections import namedtuple
from functools import partial

import pytest
import torch
from sklearn.metrics import r2_score as sk_r2score

from pytorch_lightning.metrics.regression import R2Score
from pytorch_lightning.metrics.functional import r2score
from tests.metrics.utils import BATCH_SIZE, NUM_BATCHES, MetricTester

torch.manual_seed(42)

num_targets = 5

Input = namedtuple('Input', ["preds", "target"])

_single_target_inputs = Input(preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.rand(NUM_BATCHES, BATCH_SIZE),)

_multi_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets), target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)


def _single_target_sk_metric(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        r2_score = 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


def _multi_target_sk_metric(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        r2_score = 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


@pytest.mark.parametrize("adjusted", [0, 5, 10])
@pytest.mark.parametrize("multioutput", ['raw_values', 'uniform_average', 'variance_weighted'])
@pytest.mark.parametrize(
    "preds, target, sk_metric, num_outputs",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric, 1),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric, num_targets),
    ],
)
class TestR2Score(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_r2(self, adjusted, multioutput, preds, target, sk_metric, num_outputs, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            R2Score,
            partial(sk_metric, adjusted=adjusted, multioutput=multioutput),
            dist_sync_on_step,
            metric_args=dict(adjusted=adjusted,
                             multioutput=multioutput,
                             num_outputs=num_outputs),
        )

    def test_r2_functional(self, adjusted, multioutput, preds, target, sk_metric, num_outputs):
        self.run_functional_metric_test(
            preds,
            target,
            r2score,
            partial(sk_metric, adjusted=adjusted, multioutput=multioutput),
            metric_args=dict(adjusted=adjusted,
                             multioutput=multioutput),
        )

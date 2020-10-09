import torch
import pytest
from collections import namedtuple
from functools import partial

from pytorch_lightning.metrics.regression import ExplainedVariance
from sklearn.metrics import explained_variance_score

from tests.metrics.utils import compute_batch, NUM_BATCHES, BATCH_SIZE

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


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("ddp_sync_on_step", [True, False])
@pytest.mark.parametrize("multioutput", ['raw_values', 'uniform_average', 'variance_weighted'])
@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric),
    ],
)
def test_explained_variance(ddp, ddp_sync_on_step, multioutput, preds, target, sk_metric):
    compute_batch(
        preds,
        target,
        ExplainedVariance,
        partial(sk_metric, sk_fn=partial(explained_variance_score, multioutput=multioutput)),
        ddp_sync_on_step,
        ddp,
        metric_args=dict(multioutput=multioutput),
    )

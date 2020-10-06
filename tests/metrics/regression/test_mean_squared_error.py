import torch
import pytest
from collections import namedtuple

from pytorch_lightning.metrics.regression import MeanSquaredError
from sklearn.metrics import mean_squared_error

from tests.metrics.utils import compute_batch, setup_ddp
from tests.metrics.utils import NUM_BATCHES, NUM_PROCESSES, BATCH_SIZE

num_targets = 5

Input = namedtuple('Input', ["preds", "target"])

_single_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)


def _single_target_sk_metric(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return mean_squared_error(sk_preds, sk_target)


_multi_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)


def _multi_target_sk_metric(preds, target):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    return mean_squared_error(sk_preds, sk_target)


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("ddp_sync_on_step", [True, False])
@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric),
    ],
)
def test_mean_squared_error_single(ddp, ddp_sync_on_step, preds, target, sk_metric):
    compute_batch(preds, target, MeanSquaredError, sk_metric, ddp_sync_on_step, ddp)


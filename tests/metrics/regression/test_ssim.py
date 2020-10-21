from collections import namedtuple
from functools import partial

import pytest
import torch
from skimage.metrics import structural_similarity

from pytorch_lightning.metrics.regression import SSIM
from pytorch_lightning.metrics.functional import ssim
from tests.metrics.utils import BATCH_SIZE, NUM_BATCHES, MetricTester

torch.manual_seed(42)


Input = namedtuple('Input', ["preds", "target", "multichannel"])


_inputs = []
for size, channel, coef, multichannel in [
    (16, 1, 0.9, False),
    (32, 3, 0.8, True),
    (48, 4, 0.7, True),
    (64, 5, 0.6, True),
]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size)
    _inputs.append(
        Input(
            preds=preds,
            target=preds * coef,
            multichannel=multichannel,
        )
    )


def _sk_metric(preds, target, data_range, multichannel):
    c, h, w = preds.shape[-3:]
    sk_preds = preds.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    sk_target = target.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    if not multichannel:
        sk_preds = sk_preds[:, :, :, 0]
        sk_target = sk_target[:, :, :, 0]

    return structural_similarity(
        sk_target, sk_preds, data_range=data_range, multichannel=multichannel, gaussian_weights=True, win_size=11
    )


@pytest.mark.parametrize(
    "preds, target, multichannel",
    [(i.preds, i.target, i.multichannel) for i in _inputs],
)
class TestSSIM(MetricTester):
    atol = 1e-3  # TODO: ideally tests should pass with lower tolerance

    # TODO: for some reason this test hangs with ddp=True
    # @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_ssim(self, preds, target, multichannel, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SSIM,
            partial(_sk_metric, data_range=1.0, multichannel=multichannel),
            metric_args={"data_range": 1.0},
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_ssim_functional(self, preds, target, multichannel):
        self.run_functional_metric_test(
            preds,
            target,
            ssim,
            partial(_sk_metric, data_range=1.0, multichannel=multichannel),
            metric_args={"data_range": 1.0},
        )


@pytest.mark.parametrize(
    ['pred', 'target', 'kernel', 'sigma'],
    [
        pytest.param([1, 16, 16], [1, 16, 16], [11, 11], [1.5, 1.5]),  # len(shape)
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 11], [1.5]),  # len(kernel), len(sigma)
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11], [1.5, 1.5]),  # len(kernel), len(sigma)
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11], [1.5]),  # len(kernel), len(sigma)
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 0], [1.5, 1.5]),  # invalid kernel input
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 10], [1.5, 1.5]),  # invalid kernel input
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, -11], [1.5, 1.5]),  # invalid kernel input
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 11], [1.5, 0]),  # invalid sigma input
        pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 0], [1.5, -1.5]),  # invalid sigma input
    ],
)
def test_ssim_invalid_inputs(pred, target, kernel, sigma):
    pred_t = torch.rand(pred)
    target_t = torch.rand(target, dtype=torch.float64)
    with pytest.raises(TypeError):
        ssim(pred_t, target_t)

    pred = torch.rand(pred)
    target = torch.rand(target)
    with pytest.raises(ValueError):
        ssim(pred, target, kernel, sigma)

from collections import namedtuple
from functools import partial

import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

from pytorch_lightning.metrics.regression import PSNR
from pytorch_lightning.metrics.functional import psnr
from tests.metrics.utils import BATCH_SIZE, NUM_BATCHES, MetricTester

torch.manual_seed(42)


Input = namedtuple('Input', ["preds", "target"])

_inputs = [
    Input(
        preds=torch.randint(n_cls_pred, (NUM_BATCHES, BATCH_SIZE), dtype=torch.float),
        target=torch.randint(n_cls_target, (NUM_BATCHES, BATCH_SIZE), dtype=torch.float),
    )
    for n_cls_pred, n_cls_target in [(10, 10), (5, 10), (10, 5)]
]


def _sk_metric(preds, target, data_range):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return peak_signal_noise_ratio(sk_target, sk_preds, data_range=data_range)


def _base_e_sk_metric(preds, target, data_range):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return peak_signal_noise_ratio(sk_target, sk_preds, data_range=data_range) * np.log(10)


@pytest.mark.parametrize(
    "preds, target, data_range",
    [
        (_inputs[0].preds, _inputs[0].target, 10),
        (_inputs[1].preds, _inputs[1].target, 10),
        (_inputs[2].preds, _inputs[2].target, 5),
    ],
)
@pytest.mark.parametrize(
    "base, sk_metric",
    [
        (10.0, _sk_metric),
        (2.718281828459045, _base_e_sk_metric),
    ],
)
class TestPSNR(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_psnr(self, preds, target, data_range, base, sk_metric, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PSNR,
            partial(sk_metric, data_range=data_range),
            metric_args={"data_range": data_range, "base": base},
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_psnr_functional(self, preds, target, sk_metric, data_range, base):
        self.run_functional_metric_test(
            preds,
            target,
            psnr,
            partial(sk_metric, data_range=data_range),
            metric_args={"data_range": data_range, "base": base},
        )

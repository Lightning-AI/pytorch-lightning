from collections import namedtuple
from functools import partial

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio

from pytorch_lightning.metrics.functional import psnr
from pytorch_lightning.metrics.regression import PSNR
from tests.metrics.utils import BATCH_SIZE, MetricTester, NUM_BATCHES

torch.manual_seed(42)

Input = namedtuple('Input', ["preds", "target"])

_input_size = (NUM_BATCHES, BATCH_SIZE, 32, 32)
_inputs = [
    Input(
        preds=torch.randint(n_cls_pred, _input_size, dtype=torch.float),
        target=torch.randint(n_cls_target, _input_size, dtype=torch.float),
    ) for n_cls_pred, n_cls_target in [(10, 10), (5, 10), (10, 5)]
]


def _to_sk_peak_signal_noise_ratio_inputs(value, dim):
    value = value.numpy()
    batches = value[None] if value.ndim == len(_input_size) - 1 else value

    if dim is None:
        return [batches]

    num_dims = np.size(dim)
    if not num_dims:
        return batches

    inputs = []
    for batch in batches:
        batch = np.moveaxis(batch, dim, np.arange(-num_dims, 0))
        psnr_input_shape = batch.shape[-num_dims:]
        inputs.extend(batch.reshape(-1, *psnr_input_shape))
    return inputs


def _sk_metric(preds, target, data_range, reduction, dim):
    sk_preds_lists = _to_sk_peak_signal_noise_ratio_inputs(preds, dim=dim)
    sk_target_lists = _to_sk_peak_signal_noise_ratio_inputs(target, dim=dim)
    np_reduce_map = {"elementwise_mean": np.mean, "none": np.array, "sum": np.sum}
    return np_reduce_map[reduction]([
        peak_signal_noise_ratio(sk_target, sk_preds, data_range=data_range)
        for sk_target, sk_preds in zip(sk_target_lists, sk_preds_lists)
    ])


def _base_e_sk_metric(preds, target, data_range, reduction, dim):
    return _sk_metric(preds, target, data_range, reduction, dim) * np.log(10)


@pytest.mark.parametrize(
    "preds, target, data_range, reduction, dim",
    [
        (_inputs[0].preds, _inputs[0].target, 10, "elementwise_mean", None),
        (_inputs[1].preds, _inputs[1].target, 10, "elementwise_mean", None),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", None),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", 1),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", (1, 2)),
        (_inputs[2].preds, _inputs[2].target, 5, "none", (1, 2)),
        (_inputs[2].preds, _inputs[2].target, 5, "sum", (1, 2)),
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
    def test_psnr(self, preds, target, data_range, base, reduction, dim, sk_metric, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PSNR,
            partial(sk_metric, data_range=data_range, reduction=reduction, dim=dim),
            metric_args={
                "data_range": data_range,
                "base": base,
                "reduction": reduction,
                "dim": dim
            },
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_psnr_functional(self, preds, target, sk_metric, data_range, base, reduction, dim):
        self.run_functional_metric_test(
            preds,
            target,
            psnr,
            partial(sk_metric, data_range=data_range, reduction=reduction, dim=dim),
            metric_args={
                "data_range": data_range,
                "base": base,
                "reduction": reduction,
                "dim": dim
            },
        )


def test_missing_data_range():
    with pytest.raises(ValueError):
        PSNR(data_range=None, dim=0)

    with pytest.raises(ValueError):
        psnr(_inputs[0].preds, _inputs[0].target, data_range=None, dim=0)

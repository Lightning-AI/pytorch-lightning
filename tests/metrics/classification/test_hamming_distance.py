import pytest
import torch
from sklearn.metrics import hamming_loss as sk_hamming_loss

from pytorch_lightning.metrics import HammingDistance
from pytorch_lightning.metrics.classification.helpers import _input_format_classification
from pytorch_lightning.metrics.functional import hamming_distance
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs,
    _multidim_multiclass_inputs,
    _multidim_multiclass_prob_inputs,
    _multilabel_inputs,
    _multilabel_multidim_inputs,
    _multilabel_multidim_prob_inputs,
    _multilabel_prob_inputs,
)
from tests.metrics.utils import MetricTester, THRESHOLD

torch.manual_seed(42)


def _sk_hamming_loss(preds, target):
    sk_preds, sk_target, _ = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()
    sk_preds, sk_target = sk_preds.reshape(sk_preds.shape[0], -1), sk_target.reshape(sk_target.shape[0], -1)

    return sk_hamming_loss(y_true=sk_target, y_pred=sk_preds)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target),
        (_binary_inputs.preds, _binary_inputs.target),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target),
        (_multilabel_inputs.preds, _multilabel_inputs.target),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target),
        (_multiclass_inputs.preds, _multiclass_inputs.target),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target),
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target),
        (_multilabel_multidim_prob_inputs.preds, _multilabel_multidim_prob_inputs.target),
        (_multilabel_multidim_inputs.preds, _multilabel_multidim_inputs.target),
    ],
)
class TestHammingDistance(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_hamming_distance_class(self, ddp, dist_sync_on_step, preds, target):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=HammingDistance,
            sk_metric=_sk_hamming_loss,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"threshold": THRESHOLD},
        )

    def test_hamming_distance_fn(self, preds, target):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=hamming_distance,
            sk_metric=_sk_hamming_loss,
            metric_args={"threshold": THRESHOLD},
        )


@pytest.mark.parametrize("threshold", [1.5])
def test_wrong_params(threshold):
    preds, target = _multiclass_prob_inputs.preds, _multiclass_prob_inputs.target

    with pytest.raises(ValueError):
        ham_dist = HammingDistance(threshold=threshold)
        ham_dist(preds, target)
        ham_dist.compute()

    with pytest.raises(ValueError):
        hamming_distance(preds, target, threshold=threshold)

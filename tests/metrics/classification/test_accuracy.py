import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, hamming_loss

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.classification.utils import _input_format_classification
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs,
    _multidim_multiclass_inputs,
    _multidim_multiclass_prob_inputs,
    _multilabel_inputs,
    _multilabel_prob_inputs,
)
from tests.metrics.utils import THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_accuracy(preds, target):
    sk_preds, sk_target = _input_format_classification(preds, target, THRESHOLD, None)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def test_accuracy_invalid_shape():
    with pytest.raises(ValueError):
        acc = Accuracy()
        acc.update(preds=torch.rand(1), target=torch.rand(1, 2, 3))


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("dist_sync_on_step", [False, True])
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
        # (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target),
    ],
)
class TestAccuracy(MetricTester):
    def test_accuracy(self, ddp, dist_sync_on_step, preds, target):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Accuracy,
            sk_metric=_sk_accuracy,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"threshold": THRESHOLD},
        )

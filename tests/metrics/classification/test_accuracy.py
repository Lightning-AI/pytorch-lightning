import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, hamming_loss

from pytorch_lightning.metrics import Accuracy, HammingLoss
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
    _multilabel_multidim_prob_inputs,
    _multilabel_multidim_inputs,
    _multidim_multiclass_prob_inputs1,
)
from tests.metrics.utils import THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_accuracy(preds, target):
    sk_preds, sk_target, _ = _input_format_classification(preds, target, threshold=THRESHOLD, logits=False)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()
    sk_preds, sk_target = sk_preds.reshape(sk_preds.shape[0], -1), sk_target.reshape(sk_target.shape[0], -1)

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)

def _sk_hamming_loss(preds, target):
    sk_preds, sk_target, _ = _input_format_classification(preds, target, threshold=THRESHOLD, logits=False)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()
    sk_preds, sk_target = sk_preds.reshape(sk_preds.shape[0], -1), sk_target.reshape(sk_target.shape[0], -1)

    return hamming_loss(y_true=sk_target, y_pred=sk_preds)

@pytest.mark.parametrize("metric, sk_metric", [(Accuracy, _sk_accuracy), (HammingLoss, _sk_hamming_loss)])
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
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target),
        (_multilabel_multidim_prob_inputs.preds, _multilabel_multidim_prob_inputs.target),
        (_multilabel_multidim_inputs.preds, _multilabel_multidim_inputs.target),
        (_multidim_multiclass_prob_inputs1.preds, _multidim_multiclass_prob_inputs1.target),
    ],
)
class TestAccuracy(MetricTester):
    def test_accuracy(self, ddp, dist_sync_on_step, preds, target, metric, sk_metric):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric,
            sk_metric=sk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"threshold": THRESHOLD, "logits": False},
        )

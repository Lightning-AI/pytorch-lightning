import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score

from pytorch_lightning.metrics.classification.accuracy import Accuracy
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


def _sk_accuracy_binary_prob(preds, target):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def _sk_accuracy_binary(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def _sk_accuracy_multilabel_prob(preds, target):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def _sk_accuracy_multilabel(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def _sk_accuracy_multiclass_prob(preds, target):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def _sk_accuracy_multiclass(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def _sk_accuracy_multidim_multiclass_prob(preds, target):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def _sk_accuracy_multidim_multiclass(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


def test_accuracy_invalid_shape():
    with pytest.raises(ValueError):
        acc = Accuracy()
        acc.update(preds=torch.rand(1), target=torch.rand(1, 2, 3))


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, _sk_accuracy_binary_prob),
        (_binary_inputs.preds, _binary_inputs.target, _sk_accuracy_binary),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, _sk_accuracy_multilabel_prob),
        (_multilabel_inputs.preds, _multilabel_inputs.target, _sk_accuracy_multilabel),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target, _sk_accuracy_multiclass_prob),
        (_multiclass_inputs.preds, _multiclass_inputs.target, _sk_accuracy_multiclass),
        (
            _multidim_multiclass_prob_inputs.preds,
            _multidim_multiclass_prob_inputs.target,
            _sk_accuracy_multidim_multiclass_prob,
        ),
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, _sk_accuracy_multidim_multiclass),
    ],
)
class TestAccuracy(MetricTester):
    def test_accuracy(self, ddp, dist_sync_on_step, preds, target, sk_metric):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Accuracy,
            sk_metric=sk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"threshold": THRESHOLD},
        )

from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve

from pytorch_lightning.metrics.classification.precision_recall_curve import PrecisionRecallCurve
from pytorch_lightning.metrics.functional.precision_recall_curve import precision_recall_curve
from tests.metrics.classification.inputs import (
    _binary_prob_inputs,
    _multiclass_prob_inputs,
    _multidim_multiclass_prob_inputs,
)
from tests.metrics.utils import NUM_CLASSES, MetricTester

torch.manual_seed(42)


def _binary_prob_sk_metric(preds, target, num_classes=1):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_precision_recall_curve(y_true=sk_target, probas_pred=sk_preds)


def _multiclass_prob_sk_metric(preds, target, num_classes=1):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_precision_recall_curve(y_true=sk_target, probas_pred=sk_preds)


def _multidim_multiclass_prob_sk_metric(preds, target, num_classes=1):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_precision_recall_curve(y_true=sk_target, probas_pred=sk_preds)


@pytest.mark.parametrize("preds, target, sk_metric, num_classes", [
    (_binary_prob_inputs.preds, _binary_prob_inputs.target, _binary_prob_sk_metric, 2),
    (
        _multiclass_prob_inputs.preds,
        _multiclass_prob_inputs.target,
        _multiclass_prob_sk_metric,
        NUM_CLASSES),
    (
        _multidim_multiclass_prob_inputs.preds,
        _multidim_multiclass_prob_inputs.target,
        _multidim_multiclass_prob_sk_metric,
        NUM_CLASSES
    ),
])
class TestPrecisionRecallCurve(MetricTester):
    #@pytest.mark.parametrize("ddp", [True, False])
    #@pytest.mark.parametrize("dist_sync_on_step", [True, False])
    #def test_precision_recall_curve(self, normalize, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
    #    self.run_class_metric_test(
    #                       ddp=ddp,
    #                       preds=preds,
    #                       target=target,
    #                       metric_class=PrecisionRecallCurve,
    #                       sk_metric=partial(sk_metric, num_classes=num_classes),
    #                       dist_sync_on_step=dist_sync_on_step,
    #                       metric_args={"num_classes": num_classes}
    #                       )

    def test_precision_recall_curve_functional(self, preds, target, sk_metric, num_classes):
        self.run_functional_metric_test(preds,
                                        target,
                                        metric_functional=precision_recall_curve,
                                        sk_metric=partial(sk_metric, num_classes=num_classes),
                                        metric_args={"num_classes": num_classes}
                                        )
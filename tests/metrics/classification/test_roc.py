from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_curve as sk_roc_curve

from pytorch_lightning.metrics.classification.roc import ROC
from pytorch_lightning.metrics.functional.roc import roc
from tests.metrics.classification.inputs import _input_binary_prob
from tests.metrics.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.metrics.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.metrics.utils import MetricTester, NUM_CLASSES

torch.manual_seed(42)


def _sk_roc_curve(y_true, probas_pred, num_classes=1):
    """ Adjusted comparison function that can also handles multiclass """
    if num_classes == 1:
        return sk_roc_curve(y_true, probas_pred, drop_intermediate=False)

    fpr, tpr, thresholds = [], [], []
    for i in range(num_classes):
        y_true_temp = np.zeros_like(y_true)
        y_true_temp[y_true == i] = 1
        res = sk_roc_curve(y_true_temp, probas_pred[:, i], drop_intermediate=False)
        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])
    return fpr, tpr, thresholds


def _sk_roc_binary_prob(preds, target, num_classes=1):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_roc_multiclass_prob(preds, target, num_classes=1):
    sk_preds = preds.reshape(-1, num_classes).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_roc_multidim_multiclass_prob(preds, target, num_classes=1):
    sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    sk_target = target.view(-1).numpy()
    return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes", [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_roc_binary_prob, 1),
        (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_roc_multiclass_prob, NUM_CLASSES),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_roc_multidim_multiclass_prob, NUM_CLASSES),
    ]
)
class TestROC(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_roc(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ROC,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes}
        )

    def test_roc_functional(self, preds, target, sk_metric, num_classes):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=roc,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            metric_args={"num_classes": num_classes},
        )


@pytest.mark.parametrize(['pred', 'target', 'expected_tpr', 'expected_fpr'], [
    pytest.param([0, 1], [0, 1], [0, 1, 1], [0, 0, 1]),
    pytest.param([1, 0], [0, 1], [0, 0, 1], [0, 1, 1]),
    pytest.param([1, 1], [1, 0], [0, 1], [0, 1]),
    pytest.param([1, 0], [1, 0], [0, 1, 1], [0, 0, 1]),
    pytest.param([0.5, 0.5], [0, 1], [0, 1], [0, 1]),
])
def test_roc_curve(pred, target, expected_tpr, expected_fpr):
    fpr, tpr, thresh = roc(torch.tensor(pred), torch.tensor(target))

    assert fpr.shape == tpr.shape
    assert fpr.size(0) == thresh.size(0)
    assert torch.allclose(fpr, torch.tensor(expected_fpr).to(fpr))
    assert torch.allclose(tpr, torch.tensor(expected_tpr).to(tpr))

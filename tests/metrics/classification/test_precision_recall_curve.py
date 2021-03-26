from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve

from pytorch_lightning.metrics.classification.precision_recall_curve import PrecisionRecallCurve
from pytorch_lightning.metrics.functional.precision_recall_curve import precision_recall_curve
from tests.metrics.classification.inputs import _input_binary_prob
from tests.metrics.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.metrics.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.metrics.utils import MetricTester, NUM_CLASSES

torch.manual_seed(42)


def _sk_precision_recall_curve(y_true, probas_pred, num_classes=1):
    """ Adjusted comparison function that can also handles multiclass """
    if num_classes == 1:
        return sk_precision_recall_curve(y_true, probas_pred)

    precision, recall, thresholds = [], [], []
    for i in range(num_classes):
        y_true_temp = np.zeros_like(y_true)
        y_true_temp[y_true == i] = 1
        res = sk_precision_recall_curve(y_true_temp, probas_pred[:, i])
        precision.append(res[0])
        recall.append(res[1])
        thresholds.append(res[2])
    return precision, recall, thresholds


def _sk_prec_rc_binary_prob(preds, target, num_classes=1):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_precision_recall_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_prec_rc_multiclass_prob(preds, target, num_classes=1):
    sk_preds = preds.reshape(-1, num_classes).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_precision_recall_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_prec_rc_multidim_multiclass_prob(preds, target, num_classes=1):
    sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    sk_target = target.view(-1).numpy()
    return _sk_precision_recall_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes", [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_prec_rc_binary_prob, 1),
        (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_prec_rc_multiclass_prob, NUM_CLASSES),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_prec_rc_multidim_multiclass_prob, NUM_CLASSES),
    ]
)
class TestPrecisionRecallCurve(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_precision_recall_curve(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=PrecisionRecallCurve,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes}
        )

    def test_precision_recall_curve_functional(self, preds, target, sk_metric, num_classes):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=precision_recall_curve,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            metric_args={"num_classes": num_classes},
        )


@pytest.mark.parametrize(
    ['pred', 'target', 'expected_p', 'expected_r', 'expected_t'],
    [pytest.param([1, 2, 3, 4], [1, 0, 0, 1], [0.5, 1 / 3, 0.5, 1., 1.], [1, 0.5, 0.5, 0.5, 0.], [1, 2, 3, 4])]
)
def test_pr_curve(pred, target, expected_p, expected_r, expected_t):
    p, r, t = precision_recall_curve(torch.tensor(pred), torch.tensor(target))
    assert p.size() == r.size()
    assert p.size(0) == t.size(0) + 1

    assert torch.allclose(p, torch.tensor(expected_p).to(p))
    assert torch.allclose(r, torch.tensor(expected_r).to(r))
    assert torch.allclose(t, torch.tensor(expected_t).to(t))

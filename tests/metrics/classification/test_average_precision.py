from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score as sk_average_precision_score

from pytorch_lightning.metrics.classification.average_precision import AveragePrecision
from pytorch_lightning.metrics.functional.average_precision import average_precision
from tests.metrics.classification.inputs import _input_binary_prob
from tests.metrics.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.metrics.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.metrics.utils import MetricTester, NUM_CLASSES

torch.manual_seed(42)


def _sk_average_precision_score(y_true, probas_pred, num_classes=1):
    if num_classes == 1:
        return sk_average_precision_score(y_true, probas_pred)

    res = []
    for i in range(num_classes):
        y_true_temp = np.zeros_like(y_true)
        y_true_temp[y_true == i] = 1
        res.append(sk_average_precision_score(y_true_temp, probas_pred[:, i]))
    return res


def _sk_avg_prec_binary_prob(preds, target, num_classes=1):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_average_precision_score(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_avg_prec_multiclass_prob(preds, target, num_classes=1):
    sk_preds = preds.reshape(-1, num_classes).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_average_precision_score(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_avg_prec_multidim_multiclass_prob(preds, target, num_classes=1):
    sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    sk_target = target.view(-1).numpy()
    return _sk_average_precision_score(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes", [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_avg_prec_binary_prob, 1),
        (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_avg_prec_multiclass_prob, NUM_CLASSES),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_avg_prec_multidim_multiclass_prob, NUM_CLASSES),
    ]
)
class TestAveragePrecision(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_average_precision(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=AveragePrecision,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes}
        )

    def test_average_precision_functional(self, preds, target, sk_metric, num_classes):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=average_precision,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            metric_args={"num_classes": num_classes},
        )


@pytest.mark.parametrize(
    ['scores', 'target', 'expected_score'],
    [
        # Check the average_precision_score of a constant predictor is
        # the TPR
        # Generate a dataset with 25% of positives
        # And a constant score
        # The precision is then the fraction of positive whatever the recall
        # is, as there is only one threshold:
        pytest.param(torch.tensor([1, 1, 1, 1]), torch.tensor([0, 0, 0, 1]), .25),
        # With threshold 0.8 : 1 TP and 2 TN and one FN
        pytest.param(torch.tensor([.6, .7, .8, 9]), torch.tensor([1, 0, 0, 1]), .75),
    ]
)
def test_average_precision(scores, target, expected_score):
    assert average_precision(scores, target) == expected_score

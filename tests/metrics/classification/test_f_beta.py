from functools import partial

import pytest
import torch
from sklearn.metrics import fbeta_score

from pytorch_lightning.metrics.classification.utils import _input_format_classification
from pytorch_lightning.metrics import FBeta
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs,
    _multidim_multiclass_inputs,
    _multidim_multiclass_prob_inputs,
    _multilabel_inputs,
    _multilabel_inputs_no_match,
    _multilabel_prob_inputs,
)
from tests.metrics.utils import NUM_CLASSES, THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_fbeta(preds, target, average, beta, num_classes):
    average = None if average == "none" else average
    sk_preds, sk_target = _input_format_classification(preds, target, THRESHOLD, num_classes)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average=average, beta=beta)

@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("average", ["binary"])
@pytest.mark.parametrize(
    "preds, target, num_classes",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, 1),
        (_binary_inputs.preds, _binary_inputs.target, 1),
    ],
)
@pytest.mark.parametrize(
    "metric_class, beta",
    [(FBeta, 0.5), (FBeta, 1.0)],
)
class TestPrecisionRecallBinary(MetricTester):
    def test_fbeta_binary(
        self, ddp, dist_sync_on_step, preds, target, metric_class, beta, num_classes, average
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(_sk_fbeta, average=average, beta=beta, num_classes=num_classes),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "beta": beta
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )

@pytest.mark.parametrize("ddp", [False])
@pytest.mark.parametrize("dist_sync_on_step", [False])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", "none"])
@pytest.mark.parametrize(
    "preds, target, num_classes",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, 2),
        (_binary_inputs.preds, _binary_inputs.target, 2),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, NUM_CLASSES),
        (_multilabel_inputs.preds, _multilabel_inputs.target, NUM_CLASSES),
        (_multilabel_inputs_no_match.preds, _multilabel_inputs_no_match.target, NUM_CLASSES),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target, NUM_CLASSES),
        (_multiclass_inputs.preds, _multiclass_inputs.target, NUM_CLASSES),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target, NUM_CLASSES),
        # (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, NUM_CLASSES),
    ],
)
@pytest.mark.parametrize(
    "metric_class, beta",
    [(FBeta, 0.5), (FBeta, 1.0)],
)
class TestFBeta(MetricTester):
    def test_fbeta(self, ddp, dist_sync_on_step, preds, target, metric_class, beta, num_classes, average):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(_sk_fbeta, average=average, beta=beta, num_classes=num_classes),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "beta": beta,
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
            },
            check_dist_sync_on_step=False,
            check_batch=False,
        )

from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import fbeta_score

from pytorch_lightning.metrics import Fbeta
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


def _sk_fbeta_binary_prob(preds, target, average='micro', beta=1.0):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average='binary', beta=beta)


def _sk_fbeta_binary(preds, target, average='micro', beta=1.0):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average='binary', beta=beta)


def _sk_fbeta_multilabel_prob(preds, target, average='micro', beta=1.0):
    sk_preds = (preds.view(-1, NUM_CLASSES).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1, NUM_CLASSES).numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average=average, beta=beta)


def _sk_fbeta_multilabel(preds, target, average='micro', beta=1.0):
    sk_preds = preds.view(-1, NUM_CLASSES).numpy()
    sk_target = target.view(-1, NUM_CLASSES).numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average=average, beta=beta)


def _sk_fbeta_multiclass_prob(preds, target, average='micro', beta=1.0):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average=average, beta=beta)


def _sk_fbeta_multiclass(preds, target, average='micro', beta=1.0):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average=average, beta=beta)


def _sk_fbeta_multidim_multiclass_prob(preds, target, average='micro', beta=1.0):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average=average, beta=beta)


def _sk_fbeta_multidim_multiclass(preds, target, average='micro', beta=1.0):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return fbeta_score(y_true=sk_target, y_pred=sk_preds, average=average, beta=beta)


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("average", ['micro', 'macro'])
@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes, multilabel",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, _sk_fbeta_binary_prob, 1, False),
        (_binary_inputs.preds, _binary_inputs.target, _sk_fbeta_binary, 1, False),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, _sk_fbeta_multilabel_prob, NUM_CLASSES, True),
        (_multilabel_inputs.preds, _multilabel_inputs.target, _sk_fbeta_multilabel, NUM_CLASSES, True),
        (_multilabel_inputs_no_match.preds, _multilabel_inputs_no_match.target, _sk_fbeta_multilabel, NUM_CLASSES, True),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target, _sk_fbeta_multiclass_prob, NUM_CLASSES, False),
        (_multiclass_inputs.preds, _multiclass_inputs.target, _sk_fbeta_multiclass, NUM_CLASSES, False),
        (
            _multidim_multiclass_prob_inputs.preds,
            _multidim_multiclass_prob_inputs.target,
            _sk_fbeta_multidim_multiclass_prob,
            NUM_CLASSES,
            False,
        ),
        (
            _multidim_multiclass_inputs.preds,
            _multidim_multiclass_inputs.target,
            _sk_fbeta_multidim_multiclass,
            NUM_CLASSES,
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    "metric_class, beta", [(Fbeta, 0.5), (Fbeta, 1.0)],
)
class TestFBeta(MetricTester):
    def test_fbeta(
        self, ddp, dist_sync_on_step, preds, target, sk_metric, metric_class, beta, num_classes, multilabel, average
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(sk_metric, average=average, beta=beta),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "beta": beta,
                "num_classes": num_classes,
                "average": average,
                "multilabel": multilabel,
                "threshold": THRESHOLD,
            },
            check_dist_sync_on_step=False,
            check_batch=False,
        )

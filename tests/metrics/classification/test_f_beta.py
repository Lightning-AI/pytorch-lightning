from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import fbeta_score

from pytorch_lightning.metrics import FBeta
from pytorch_lightning.metrics.functional import fbeta, f1
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
@pytest.mark.parametrize("average", ['micro', 'macro', 'weighted', None])
@pytest.mark.parametrize("beta", [0.5, 1.0])
class TestFBeta(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_fbeta(
        self, preds, target, sk_metric, num_classes, multilabel, average, beta, ddp, dist_sync_on_step
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=FBeta,
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

    def test_fbeta_functional(
        self, preds, target, sk_metric, num_classes, multilabel, average, beta
    ):
        self.run_functional_metric_test(preds=preds,
                                        target=target,
                                        metric_functional=fbeta,
                                        sk_metric=partial(sk_metric, average=average, beta=beta),
                                        metric_args={
                                            "beta": beta,
                                            "num_classes": num_classes,
                                            "average": average,
                                            "multilabel": multilabel,
                                            "threshold": THRESHOLD}
                                        )


@pytest.mark.parametrize(['pred', 'target', 'beta', 'exp_score'], [
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], 0.5, [0.5, 0.5]),
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], 1, [0.5, 0.5]),
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], 2, [0.5, 0.5]),
])
def test_fbeta_score(pred, target, beta, exp_score):
    score = fbeta(torch.tensor(pred), torch.tensor(target), num_classes=1, beta=beta, average='none')
    assert torch.allclose(score, torch.tensor(exp_score))


@pytest.mark.parametrize(['pred', 'target', 'exp_score'], [
    pytest.param([0., 0., 0., 0.], [1., 1., 1., 1.], [0.0, 0.0]),
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], [0.5, 0.5]),
    pytest.param([1., 0., 1., 0.], [1., 0., 1., 0.], [1.0, 1.0]),
])
def test_f1_score(pred, target, exp_score):
    score = f1(torch.tensor(pred), torch.tensor(target), num_classes=1, average='none')
    assert torch.allclose(score, torch.tensor(exp_score))

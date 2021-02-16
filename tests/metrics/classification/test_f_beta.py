from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import fbeta_score

from pytorch_lightning.metrics import F1, FBeta
from pytorch_lightning.metrics.functional import f1, fbeta
from tests.metrics.classification.inputs import _input_binary, _input_binary_prob
from tests.metrics.classification.inputs import _input_multiclass as _input_mcls
from tests.metrics.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.metrics.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.metrics.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.metrics.classification.inputs import _input_multilabel as _input_mlb
from tests.metrics.classification.inputs import _input_multilabel_no_match as _input_mlb_nomatch
from tests.metrics.classification.inputs import _input_multilabel_prob as _mlb_prob_inputs
from tests.metrics.utils import MetricTester, NUM_CLASSES, THRESHOLD

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
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_fbeta_binary_prob, 1, False),
        (_input_binary.preds, _input_binary.target, _sk_fbeta_binary, 1, False),
        (_mlb_prob_inputs.preds, _mlb_prob_inputs.target, _sk_fbeta_multilabel_prob, NUM_CLASSES, True),
        (_input_mlb.preds, _input_mlb.target, _sk_fbeta_multilabel, NUM_CLASSES, True),
        (_input_mlb_nomatch.preds, _input_mlb_nomatch.target, _sk_fbeta_multilabel, NUM_CLASSES, True),
        (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_fbeta_multiclass_prob, NUM_CLASSES, False),
        (_input_mcls.preds, _input_mcls.target, _sk_fbeta_multiclass, NUM_CLASSES, False),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_fbeta_multidim_multiclass_prob, NUM_CLASSES, False),
        (_input_mdmc.preds, _input_mdmc.target, _sk_fbeta_multidim_multiclass, NUM_CLASSES, False),
    ],
)
@pytest.mark.parametrize("average", ['micro', 'macro', 'weighted', None])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
class TestFBeta(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_fbeta(self, preds, target, sk_metric, num_classes, multilabel, average, beta, ddp, dist_sync_on_step):
        metric_class = F1 if beta == 1.0 else partial(FBeta, beta=beta)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(sk_metric, average=average, beta=beta),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "multilabel": multilabel,
                "threshold": THRESHOLD,
            },
            check_dist_sync_on_step=False,
            check_batch=False,
        )

    def test_fbeta_functional(self, preds, target, sk_metric, num_classes, multilabel, average, beta):
        metric_functional = f1 if beta == 1.0 else partial(fbeta, beta=beta)

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            sk_metric=partial(sk_metric, average=average, beta=beta),
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "multilabel": multilabel,
                "threshold": THRESHOLD
            }
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

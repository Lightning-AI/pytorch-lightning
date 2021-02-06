from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from pytorch_lightning.metrics.classification.confusion_matrix import ConfusionMatrix
from pytorch_lightning.metrics.functional.confusion_matrix import confusion_matrix
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _mclass_inputs,
    _mclass_prob_inputs,
    _mdim_mclass_inputs,
    _mdim_mclass_prob_inputs,
    _mlabel_inputs,
    _mlabel_prob_inputs,
)
from tests.metrics.utils import MetricTester, NUM_CLASSES, THRESHOLD

torch.manual_seed(42)


def _binary_prob_sk_metric(preds, target, normalize=None):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


def _binary_sk_metric(preds, target, normalize=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


def _mlabel_prob_sk_metric(preds, target, normalize=None):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


def _mlabel_sk_metric(preds, target, normalize=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


def _mclass_prob_sk_metric(preds, target, normalize=None):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


def _mclass_sk_metric(preds, target, normalize=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


def _mdim_mclass_prob_sk_metric(preds, target, normalize=None):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


def _mdim_mclass_sk_metric(preds, target, normalize=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


@pytest.mark.parametrize("normalize", ['true', 'pred', 'all', None])
@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [(_binary_prob_inputs.preds, _binary_prob_inputs.target, _binary_prob_sk_metric, 2),
     (_binary_inputs.preds, _binary_inputs.target, _binary_sk_metric, 2),
     (_mlabel_prob_inputs.preds, _mlabel_prob_inputs.target, _mlabel_prob_sk_metric, 2),
     (_mlabel_inputs.preds, _mlabel_inputs.target, _mlabel_sk_metric, 2),
     (_mclass_prob_inputs.preds, _mclass_prob_inputs.target, _mclass_prob_sk_metric, NUM_CLASSES),
     (_mclass_inputs.preds, _mclass_inputs.target, _mclass_sk_metric, NUM_CLASSES),
     (_mdim_mclass_prob_inputs.preds, _mdim_mclass_prob_inputs.target, _mdim_mclass_prob_sk_metric, NUM_CLASSES),
     (_mdim_mclass_inputs.preds, _mdim_mclass_inputs.target, _mdim_mclass_sk_metric, NUM_CLASSES)]
)
class TestConfusionMatrix(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_confusion_matrix(self, normalize, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ConfusionMatrix,
            sk_metric=partial(sk_metric, normalize=normalize),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "threshold": THRESHOLD,
                "normalize": normalize
            }
        )

    def test_confusion_matrix_functional(self, normalize, preds, target, sk_metric, num_classes):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=confusion_matrix,
            sk_metric=partial(sk_metric, normalize=normalize),
            metric_args={
                "num_classes": num_classes,
                "threshold": THRESHOLD,
                "normalize": normalize
            }
        )


def test_warning_on_nan(tmpdir):
    preds = torch.randint(3, size=(20, ))
    target = torch.randint(3, size=(20, ))

    with pytest.warns(UserWarning, match='.* nan values found in confusion matrix have been replaced with zeros.'):
        confusion_matrix(preds, target, num_classes=5, normalize='true')

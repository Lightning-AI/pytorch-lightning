from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from pytorch_lightning.metrics.classification.confusion_matrix import ConfusionMatrix
from pytorch_lightning.metrics.functional.confusion_matrix import confusion_matrix
from pytorch_lightning.metrics.classification.utils import _input_format_classification
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
from tests.metrics.utils import NUM_CLASSES, THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_confmat(preds, target, normalize, num_classes):
    sk_preds, sk_target, _ = _input_format_classification(
        preds, target, THRESHOLD, num_classes=num_classes, is_multiclass=True
    )

    if len(sk_preds.shape) > 2:
        sk_preds = torch.movedim(sk_preds, 1, -1).reshape(-1, num_classes)
        sk_target = torch.movedim(sk_target, 1, -1).reshape(-1, num_classes)

    sk_preds = sk_preds.argmax(dim=1).numpy()
    sk_target = sk_target.argmax(dim=1).numpy()

    return sk_confusion_matrix(y_true=sk_target, y_pred=sk_preds, normalize=normalize)


@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
@pytest.mark.parametrize(
    "preds, target, num_classes",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, 2),
        (_binary_inputs.preds, _binary_inputs.target, 2),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, 2),
        (_multilabel_inputs.preds, _multilabel_inputs.target, 2),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target, NUM_CLASSES),
        (_multiclass_inputs.preds, _multiclass_inputs.target, NUM_CLASSES),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target, NUM_CLASSES),
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, NUM_CLASSES),
    ],
)
class TestConfusionMatrix(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_confusion_matrix(self, normalize, preds, target, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ConfusionMatrix,
            sk_metric=partial(_sk_confmat, normalize=normalize, num_classes=num_classes),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "normalize": normalize},
        )

    def test_confusion_matrix_functional(self, normalize, preds, target, num_classes):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=confusion_matrix,
            sk_metric=partial(_sk_confmat, normalize=normalize, num_classes=num_classes),
            metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "normalize": normalize},
        )


def test_warning_on_nan(tmpdir):
    preds = torch.randint(3, size=(20,))
    target = torch.randint(3, size=(20,))

    with pytest.warns(UserWarning, match=".* nan values found in confusion matrix have been replaced with zeros."):
        confmat = confusion_matrix(preds, target, num_classes=5, normalize="true")

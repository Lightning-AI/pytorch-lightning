from functools import partial

import numpy as np
import pytest
import torch
from scipy.spatial.distance import dice as sk_dice

from pytorch_lightning.metrics.functional import dice_score
from pytorch_lightning.metrics.classification.helpers import _input_format_classification
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs,
    _multidim_multiclass_inputs,
    _multidim_multiclass_prob_inputs,
    _multilabel_inputs,
    _multilabel_prob_inputs,
    _multilabel_multidim_prob_inputs,
    _multilabel_multidim_inputs,
)
from tests.metrics.utils import THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_dice(preds, target):
    sk_preds, sk_target, mode = _input_format_classification(preds, target)
    sk_preds, sk_target = sk_preds.numpy().flatten(), sk_target.numpy().flatten()

    return sk_dice(u=sk_target, v=sk_preds)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target),
        (_binary_inputs.preds, _binary_inputs.target),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target),
        (_multilabel_inputs.preds, _multilabel_inputs.target),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target),
        (_multiclass_inputs.preds, _multiclass_inputs.target),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target),
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target),
        (_multilabel_multidim_prob_inputs.preds, _multilabel_multidim_prob_inputs.target),
        (_multilabel_multidim_inputs.preds, _multilabel_multidim_inputs.target),
    ],
)
class TestAccuracies(MetricTester):
    # @pytest.mark.parametrize("ddp", [False, True])
    # @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    # def test_dice_class(self, ddp, dist_sync_on_step, preds, target, subset_accuracy):
    #     self.run_class_metric_test(
    #         ddp=ddp,
    #         preds=preds,
    #         target=target,
    #         metric_class=Accuracy,
    #         sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
    #         dist_sync_on_step=dist_sync_on_step,
    #     )

    def test_dice_fn(self, preds, target):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=dice_score,
            sk_metric=_sk_dice,
        )

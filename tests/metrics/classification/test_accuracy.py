from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score as sk_accuracy

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.classification.helpers import _input_format_classification, DataType
from pytorch_lightning.metrics.functional import accuracy
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs,
    _multidim_multiclass_inputs,
    _multidim_multiclass_prob_inputs,
    _multilabel_inputs,
    _multilabel_multidim_inputs,
    _multilabel_multidim_prob_inputs,
    _multilabel_prob_inputs,
)
from tests.metrics.utils import MetricTester, THRESHOLD

torch.manual_seed(42)


def _sk_accuracy(preds, target, subset_accuracy):
    sk_preds, sk_target, mode = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    if mode == DataType.MULTIDIM_MULTICLASS and not subset_accuracy:
        sk_preds, sk_target = np.transpose(sk_preds, (0, 2, 1)), np.transpose(sk_target, (0, 2, 1))
        sk_preds, sk_target = sk_preds.reshape(-1, sk_preds.shape[2]), sk_target.reshape(-1, sk_target.shape[2])
    elif mode == DataType.MULTIDIM_MULTICLASS and subset_accuracy:
        return np.all(sk_preds == sk_target, axis=(1, 2)).mean()
    elif mode == DataType.MULTILABEL and not subset_accuracy:
        sk_preds, sk_target = sk_preds.reshape(-1), sk_target.reshape(-1)

    return sk_accuracy(y_true=sk_target, y_pred=sk_preds)


@pytest.mark.parametrize(
    "preds, target, subset_accuracy",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, False),
        (_binary_inputs.preds, _binary_inputs.target, False),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, True),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, False),
        (_multilabel_inputs.preds, _multilabel_inputs.target, True),
        (_multilabel_inputs.preds, _multilabel_inputs.target, False),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target, False),
        (_multiclass_inputs.preds, _multiclass_inputs.target, False),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target, False),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target, True),
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, False),
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, True),
        (_multilabel_multidim_prob_inputs.preds, _multilabel_multidim_prob_inputs.target, True),
        (_multilabel_multidim_prob_inputs.preds, _multilabel_multidim_prob_inputs.target, False),
        (_multilabel_multidim_inputs.preds, _multilabel_multidim_inputs.target, True),
        (_multilabel_multidim_inputs.preds, _multilabel_multidim_inputs.target, False),
    ],
)
class TestAccuracies(MetricTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_accuracy_class(self, ddp, dist_sync_on_step, preds, target, subset_accuracy):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Accuracy,
            sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"threshold": THRESHOLD, "subset_accuracy": subset_accuracy},
        )

    def test_accuracy_fn(self, preds, target, subset_accuracy):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=accuracy,
            sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
            metric_args={"threshold": THRESHOLD, "subset_accuracy": subset_accuracy},
        )


_l1to4 = [0.1, 0.2, 0.3, 0.4]
_l1to4t3 = np.array([_l1to4, _l1to4, _l1to4])
_l1to4t3_mc = [_l1to4t3.T, _l1to4t3.T, _l1to4t3.T]

# The preds in these examples always put highest probability on class 3, second highest on class 2,
# third highest on class 1, and lowest on class 0
_topk_preds_mc = torch.tensor([_l1to4t3, _l1to4t3]).float()
_topk_target_mc = torch.tensor([[1, 2, 3], [2, 1, 0]])

# This is like for MC case, but one sample in each batch is sabotaged with 0 class prediction :)
_topk_preds_mdmc = torch.tensor([_l1to4t3_mc, _l1to4t3_mc]).float()
_topk_target_mdmc = torch.tensor([[[1, 1, 0], [2, 2, 2], [3, 3, 3]], [[2, 2, 0], [1, 1, 1], [0, 0, 0]]])


# Replace with a proper sk_metric test once sklearn 0.24 hits :)
@pytest.mark.parametrize(
    "preds, target, exp_result, k, subset_accuracy",
    [
        (_topk_preds_mc, _topk_target_mc, 1 / 6, 1, False),
        (_topk_preds_mc, _topk_target_mc, 3 / 6, 2, False),
        (_topk_preds_mc, _topk_target_mc, 5 / 6, 3, False),
        (_topk_preds_mc, _topk_target_mc, 1 / 6, 1, True),
        (_topk_preds_mc, _topk_target_mc, 3 / 6, 2, True),
        (_topk_preds_mc, _topk_target_mc, 5 / 6, 3, True),
        (_topk_preds_mdmc, _topk_target_mdmc, 1 / 6, 1, False),
        (_topk_preds_mdmc, _topk_target_mdmc, 8 / 18, 2, False),
        (_topk_preds_mdmc, _topk_target_mdmc, 13 / 18, 3, False),
        (_topk_preds_mdmc, _topk_target_mdmc, 1 / 6, 1, True),
        (_topk_preds_mdmc, _topk_target_mdmc, 2 / 6, 2, True),
        (_topk_preds_mdmc, _topk_target_mdmc, 3 / 6, 3, True),
    ],
)
def test_topk_accuracy(preds, target, exp_result, k, subset_accuracy):
    topk = Accuracy(top_k=k, subset_accuracy=subset_accuracy)

    for batch in range(preds.shape[0]):
        topk(preds[batch], target[batch])

    assert topk.compute() == exp_result

    # Test functional
    total_samples = target.shape[0] * target.shape[1]

    preds = preds.view(total_samples, 4, -1)
    target = target.view(total_samples, -1)

    assert accuracy(preds, target, top_k=k, subset_accuracy=subset_accuracy) == exp_result


# Only MC and MDMC with probs input type should be accepted for top_k
@pytest.mark.parametrize(
    "preds, target",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target),
        (_binary_inputs.preds, _binary_inputs.target),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target),
        (_multilabel_inputs.preds, _multilabel_inputs.target),
        (_multiclass_inputs.preds, _multiclass_inputs.target),
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target),
        (_multilabel_multidim_prob_inputs.preds, _multilabel_multidim_prob_inputs.target),
        (_multilabel_multidim_inputs.preds, _multilabel_multidim_inputs.target),
    ],
)
def test_topk_accuracy_wrong_input_types(preds, target):
    topk = Accuracy(top_k=1)

    with pytest.raises(ValueError):
        topk(preds[0], target[0])

    with pytest.raises(ValueError):
        accuracy(preds[0], target[0], top_k=1)


@pytest.mark.parametrize("top_k, threshold", [(0, 0.5), (None, 1.5)])
def test_wrong_params(top_k, threshold):
    preds, target = _multiclass_prob_inputs.preds, _multiclass_prob_inputs.target

    with pytest.raises(ValueError):
        acc = Accuracy(threshold=threshold, top_k=top_k)
        acc(preds, target)
        acc.compute()

    with pytest.raises(ValueError):
        accuracy(preds, target, threshold=threshold, top_k=top_k)

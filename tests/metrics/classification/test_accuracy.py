import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score as sk_accuracy, hamming_loss as sk_hamming_loss

from pytorch_lightning.metrics import Accuracy, HammingLoss
from pytorch_lightning.metrics.functional import accuracy, hamming_loss
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
    _multilabel_multidim_prob_inputs,
    _multilabel_multidim_inputs,
)
from tests.metrics.utils import THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_accuracy(preds, target):
    sk_preds, sk_target, _ = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()
    sk_preds, sk_target = sk_preds.reshape(sk_preds.shape[0], -1), sk_target.reshape(sk_target.shape[0], -1)

    return sk_accuracy(y_true=sk_target, y_pred=sk_preds)


def _sk_hamming_loss(preds, target):
    sk_preds, sk_target, _ = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()
    sk_preds, sk_target = sk_preds.reshape(sk_preds.shape[0], -1), sk_target.reshape(sk_target.shape[0], -1)

    return sk_hamming_loss(y_true=sk_target, y_pred=sk_preds)


@pytest.mark.parametrize(
    "metric, fn_metric, sk_metric", [(Accuracy, accuracy, _sk_accuracy), (HammingLoss, hamming_loss, _sk_hamming_loss)]
)
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
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_accuracy_class(self, ddp, dist_sync_on_step, preds, target, metric, sk_metric, fn_metric):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric,
            sk_metric=sk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"threshold": THRESHOLD},
        )

    def test_accuracy_fn(self, preds, target, metric, sk_metric, fn_metric):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=fn_metric,
            sk_metric=sk_metric,
            metric_args={"threshold": THRESHOLD},
        )


l1to4 = [.1, .2, .3, .4]
l1to4t3 = np.array([l1to4, l1to4, l1to4])
l1to4t3_mc = [l1to4t3.T, l1to4t3.T, l1to4t3.T]

# The preds in these examples always put highest probability on class 3, second highest on class 2,
# third highest on class 1, and lowest on class 0
topk_preds_mc = torch.tensor([l1to4t3, l1to4t3]).float()
topk_target_mc = torch.tensor([[1, 2, 3], [2, 1, 0]])

# This is like for MC case, but one sample in each batch is sabotaged with 0 class prediction :)
topk_preds_mdmc = torch.tensor([l1to4t3_mc, l1to4t3_mc]).float()
topk_target_mdmc = torch.tensor([[[1, 1, 0], [2, 2, 2], [3, 3, 3]], [[2, 2, 0], [1, 1, 1], [0, 0, 0]]])


# Replace with a proper sk_metric test once sklearn 0.24 hits :)
@pytest.mark.parametrize(
    "preds, target, exp_result, k, mdmc_accuracy",
    [
        (topk_preds_mc, topk_target_mc, 1 / 6, 1, "global"),
        (topk_preds_mc, topk_target_mc, 3 / 6, 2, "global"),
        (topk_preds_mc, topk_target_mc, 5 / 6, 3, "global"),
        (topk_preds_mc, topk_target_mc, 1 / 6, 1, "subset"),
        (topk_preds_mc, topk_target_mc, 3 / 6, 2, "subset"),
        (topk_preds_mc, topk_target_mc, 5 / 6, 3, "subset"),
        (topk_preds_mdmc, topk_target_mdmc, 1 / 6, 1, "global"),
        (topk_preds_mdmc, topk_target_mdmc, 8 / 18, 2, "global"),
        (topk_preds_mdmc, topk_target_mdmc, 13 / 18, 3, "global"),
        (topk_preds_mdmc, topk_target_mdmc, 1 / 6, 1, "subset"),
        (topk_preds_mdmc, topk_target_mdmc, 2 / 6, 2, "subset"),
        (topk_preds_mdmc, topk_target_mdmc, 3 / 6, 3, "subset"),
    ],
)
def test_topk_accuracy(preds, target, exp_result, k, mdmc_accuracy):
    topk = Accuracy(top_k=k, mdmc_accuracy=mdmc_accuracy)

    for batch in range(preds.shape[0]):
        topk(preds[batch], target[batch])

    assert topk.compute() == exp_result

    # Test functional
    total_samples = target.shape[0] * target.shape[1]

    preds = preds.view(total_samples, 4, -1)
    target = target.view(total_samples, -1)

    assert accuracy(preds, target, top_k=k, mdmc_accuracy=mdmc_accuracy) == exp_result


# Only MC and MDMC with probs input type should be accepted
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
    topk = Accuracy(top_k=2)

    with pytest.raises(ValueError):
        topk(preds[0], target[0])

    with pytest.raises(ValueError):
        accuracy(preds[0], target[0], top_k=2)

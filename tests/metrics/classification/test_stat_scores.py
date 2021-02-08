from functools import partial
from typing import Callable, Optional

import numpy as np
import pytest
import torch
from sklearn.metrics import multilabel_confusion_matrix

from pytorch_lightning.metrics import StatScores
from pytorch_lightning.metrics.classification.helpers import _input_format_classification
from pytorch_lightning.metrics.functional import stat_scores
from tests.metrics.classification.inputs import _input_binary, _input_binary_prob, _input_multiclass
from tests.metrics.classification.inputs import _input_multiclass_prob as _input_mccls_prob
from tests.metrics.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.metrics.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.metrics.classification.inputs import _input_multilabel as _input_mcls
from tests.metrics.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.metrics.utils import MetricTester, NUM_CLASSES, THRESHOLD

torch.manual_seed(42)


def _sk_stat_scores(preds, target, reduce, num_classes, is_multiclass, ignore_index, top_k, mdmc_reduce=None):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass, top_k=top_k
    )
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if reduce != "macro" and ignore_index is not None and preds.shape[1] > 1:
        sk_preds = np.delete(sk_preds, ignore_index, 1)
        sk_target = np.delete(sk_target, ignore_index, 1)

    if preds.shape[1] == 1 and reduce == "samples":
        sk_target = sk_target.T
        sk_preds = sk_preds.T

    sk_stats = multilabel_confusion_matrix(
        sk_target, sk_preds, samplewise=(reduce == "samples") and preds.shape[1] != 1
    )

    if preds.shape[1] == 1 and reduce != "samples":
        sk_stats = sk_stats[[1]].reshape(-1, 4)[:, [3, 1, 0, 2]]
    else:
        sk_stats = sk_stats.reshape(-1, 4)[:, [3, 1, 0, 2]]

    if reduce == "micro":
        sk_stats = sk_stats.sum(axis=0, keepdims=True)

    sk_stats = np.concatenate([sk_stats, sk_stats[:, [3]] + sk_stats[:, [0]]], 1)

    if reduce == "micro":
        sk_stats = sk_stats[0]

    if reduce == "macro" and ignore_index is not None and preds.shape[1]:
        sk_stats[ignore_index, :] = -1

    return sk_stats


def _sk_stat_scores_mdim_mcls(preds, target, reduce, mdmc_reduce, num_classes, is_multiclass, ignore_index, top_k):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass, top_k=top_k
    )

    if mdmc_reduce == "global":
        preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
        target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])

        return _sk_stat_scores(preds, target, reduce, None, False, ignore_index, top_k)
    elif mdmc_reduce == "samplewise":
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_stat_scores(pred_i, target_i, reduce, None, False, ignore_index, top_k)

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores)


@pytest.mark.parametrize(
    "reduce, mdmc_reduce, num_classes, inputs, ignore_index",
    [
        ["unknown", None, None, _input_binary, None],
        ["micro", "unknown", None, _input_binary, None],
        ["macro", None, None, _input_binary, None],
        ["micro", None, None, _input_mdmc_prob, None],
        ["micro", None, None, _input_binary_prob, 0],
        ["micro", None, None, _input_mccls_prob, NUM_CLASSES],
        ["micro", None, NUM_CLASSES, _input_mccls_prob, NUM_CLASSES],
    ],
)
def test_wrong_params(reduce, mdmc_reduce, num_classes, inputs, ignore_index):
    """Test a combination of parameters that are invalid and should raise an error.

    This includes invalid ``reduce`` and ``mdmc_reduce`` parameter values, not setting
    ``num_classes`` when ``reduce='macro'`, not setting ``mdmc_reduce`` when inputs
    are multi-dim multi-class``, setting ``ignore_index`` when inputs are binary, as well
    as setting ``ignore_index`` to a value higher than the number of classes.
    """
    with pytest.raises(ValueError):
        stat_scores(
            inputs.preds[0], inputs.target[0], reduce, mdmc_reduce, num_classes=num_classes, ignore_index=ignore_index
        )

    with pytest.raises(ValueError):
        sts = StatScores(reduce=reduce, mdmc_reduce=mdmc_reduce, num_classes=num_classes, ignore_index=ignore_index)
        sts(inputs.preds[0], inputs.target[0])


def test_wrong_threshold():
    with pytest.raises(ValueError):
        StatScores(threshold=1.5)


@pytest.mark.parametrize("ignore_index", [None, 0])
@pytest.mark.parametrize("reduce", ["micro", "macro", "samples"])
@pytest.mark.parametrize(
    "preds, target, sk_fn, mdmc_reduce, num_classes, is_multiclass, top_k",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_stat_scores, None, 1, None, None),
        (_input_binary.preds, _input_binary.target, _sk_stat_scores, None, 1, False, None),
        (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, None),
        (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, 2),
        (_input_mcls.preds, _input_mcls.target, _sk_stat_scores, None, NUM_CLASSES, False, None),
        (_input_mccls_prob.preds, _input_mccls_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, None),
        (_input_mccls_prob.preds, _input_mccls_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, 2),
        (_input_multiclass.preds, _input_multiclass.target, _sk_stat_scores, None, NUM_CLASSES, None, None),
        (_input_mdmc.preds, _input_mdmc.target, _sk_stat_scores_mdim_mcls, "samplewise", NUM_CLASSES, None, None),
        (
            _input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_stat_scores_mdim_mcls, "samplewise", NUM_CLASSES, None,
            None
        ),
        (_input_mdmc.preds, _input_mdmc.target, _sk_stat_scores_mdim_mcls, "global", NUM_CLASSES, None, None),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_stat_scores_mdim_mcls, "global", NUM_CLASSES, None, None),
    ],
)
class TestStatScores(MetricTester):
    # DDP tests temporarily disabled due to hanging issues
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_stat_scores_class(
        self,
        ddp: bool,
        dist_sync_on_step: bool,
        sk_fn: Callable,
        preds: torch.Tensor,
        target: torch.Tensor,
        reduce: str,
        mdmc_reduce: Optional[str],
        num_classes: Optional[int],
        is_multiclass: Optional[bool],
        ignore_index: Optional[int],
        top_k: Optional[int],
    ):
        if ignore_index is not None and preds.ndim == 2:
            pytest.skip("Skipping ignore_index test with binary inputs.")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=StatScores,
            sk_metric=partial(
                sk_fn,
                reduce=reduce,
                mdmc_reduce=mdmc_reduce,
                num_classes=num_classes,
                is_multiclass=is_multiclass,
                ignore_index=ignore_index,
                top_k=top_k,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "reduce": reduce,
                "mdmc_reduce": mdmc_reduce,
                "threshold": THRESHOLD,
                "is_multiclass": is_multiclass,
                "ignore_index": ignore_index,
                "top_k": top_k,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )

    def test_stat_scores_fn(
        self,
        sk_fn: Callable,
        preds: torch.Tensor,
        target: torch.Tensor,
        reduce: str,
        mdmc_reduce: Optional[str],
        num_classes: Optional[int],
        is_multiclass: Optional[bool],
        ignore_index: Optional[int],
        top_k: Optional[int],
    ):
        if ignore_index is not None and preds.ndim == 2:
            pytest.skip("Skipping ignore_index test with binary inputs.")

        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=stat_scores,
            sk_metric=partial(
                sk_fn,
                reduce=reduce,
                mdmc_reduce=mdmc_reduce,
                num_classes=num_classes,
                is_multiclass=is_multiclass,
                ignore_index=ignore_index,
                top_k=top_k,
            ),
            metric_args={
                "num_classes": num_classes,
                "reduce": reduce,
                "mdmc_reduce": mdmc_reduce,
                "threshold": THRESHOLD,
                "is_multiclass": is_multiclass,
                "ignore_index": ignore_index,
                "top_k": top_k,
            },
        )


_mc_k_target = torch.tensor([0, 1, 2])
_mc_k_preds = torch.tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
_ml_k_target = torch.tensor([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
_ml_k_preds = torch.tensor([[0.9, 0.2, 0.75], [0.1, 0.7, 0.8], [0.6, 0.1, 0.7]])


@pytest.mark.parametrize(
    "k, preds, target, reduce, expected",
    [
        (1, _mc_k_preds, _mc_k_target, "micro", torch.tensor([2, 1, 5, 1, 3])),
        (2, _mc_k_preds, _mc_k_target, "micro", torch.tensor([3, 3, 3, 0, 3])),
        (1, _ml_k_preds, _ml_k_target, "micro", torch.tensor([0, 3, 3, 3, 3])),
        (2, _ml_k_preds, _ml_k_target, "micro", torch.tensor([1, 5, 1, 2, 3])),
        (1, _mc_k_preds, _mc_k_target, "macro", torch.tensor([[0, 1, 1], [0, 1, 0], [2, 1, 2], [1, 0, 0], [1, 1, 1]])),
        (2, _mc_k_preds, _mc_k_target, "macro", torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1]])),
        (1, _ml_k_preds, _ml_k_target, "macro", torch.tensor([[0, 0, 0], [1, 0, 2], [1, 1, 1], [1, 2, 0], [1, 2, 0]])),
        (2, _ml_k_preds, _ml_k_target, "macro", torch.tensor([[0, 1, 0], [2, 0, 3], [0, 1, 0], [1, 1, 0], [1, 2, 0]])),
    ],
)
def test_top_k(k: int, preds: torch.Tensor, target: torch.Tensor, reduce: str, expected: torch.Tensor):
    """ A simple test to check that top_k works as expected """

    class_metric = StatScores(top_k=k, reduce=reduce, num_classes=3)
    class_metric.update(preds, target)

    assert torch.equal(class_metric.compute(), expected.T)
    assert torch.equal(stat_scores(preds, target, top_k=k, reduce=reduce, num_classes=3), expected.T)

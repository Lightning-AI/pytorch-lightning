from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import multilabel_confusion_matrix

from pytorch_lightning.metrics.classification.helpers import _input_format_classification
from pytorch_lightning.metrics import StatScores
from pytorch_lightning.metrics.functional import stat_scores
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs as _mc_prob,
    _multilabel_inputs,
    _multilabel_prob_inputs as _ml_prob,
    _multidim_multiclass_inputs as _mdmc,
    _multidim_multiclass_prob_inputs as _mdmc_prob,
)
from tests.metrics.utils import NUM_CLASSES, THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_stat_scores(preds, target, reduce, num_classes, is_multiclass, ignore_index, top_k, mdmc_reduce=None):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass, top_k=top_k
    )
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if reduce != "macro" and ignore_index and preds.shape[1] > 1:
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

    if reduce == "macro" and ignore_index and preds.shape[1]:
        sk_stats[ignore_index, :] = -1

    return sk_stats


def _sk_stat_scores_mdmc(preds, target, reduce, mdmc_reduce, num_classes, is_multiclass, ignore_index, top_k):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass, top_k=top_k
    )

    if mdmc_reduce == "global":
        shape_permute = list(range(preds.ndim))
        shape_permute[1] = shape_permute[-1]
        shape_permute[2:] = range(1, len(shape_permute) - 1)

        preds = preds.permute(*shape_permute).reshape(-1, preds.shape[1])
        target = target.permute(*shape_permute).reshape(-1, target.shape[1])

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
        ["unknown", None, None, _binary_inputs, None],
        ["micro", "unknown", None, _binary_inputs, None],
        ["macro", None, None, _binary_inputs, None],
        ["micro", None, None, _mdmc_prob, None],
        ["micro", None, None, _binary_prob_inputs, 0],
        ["micro", None, None, _mc_prob, NUM_CLASSES],
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


@pytest.mark.parametrize("ignore_index", [None, 1])
@pytest.mark.parametrize("reduce", ["micro", "macro", "samples"])
@pytest.mark.parametrize(
    "preds, target, sk_fn, mdmc_reduce, num_classes, is_multiclass, top_k",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, _sk_stat_scores, None, 1, None, None),
        (_binary_inputs.preds, _binary_inputs.target, _sk_stat_scores, None, 1, False, None),
        (_ml_prob.preds, _ml_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, None),
        (_ml_prob.preds, _ml_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, 2),
        (_multilabel_inputs.preds, _multilabel_inputs.target, _sk_stat_scores, None, NUM_CLASSES, False, None),
        (_mc_prob.preds, _mc_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, None),
        (_mc_prob.preds, _mc_prob.target, _sk_stat_scores, None, NUM_CLASSES, None, 2),
        (_multiclass_inputs.preds, _multiclass_inputs.target, _sk_stat_scores, None, NUM_CLASSES, None, None),
        (_mdmc.preds, _mdmc.target, _sk_stat_scores_mdmc, "samplewise", NUM_CLASSES, None, None),
        (_mdmc_prob.preds, _mdmc_prob.target, _sk_stat_scores_mdmc, "samplewise", NUM_CLASSES, None, None),
        (_mdmc.preds, _mdmc.target, _sk_stat_scores_mdmc, "global", NUM_CLASSES, None, None),
        (_mdmc_prob.preds, _mdmc_prob.target, _sk_stat_scores_mdmc, "global", NUM_CLASSES, None, None),
    ],
)
class TestStatScores(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_stat_scores_class(
        self,
        ddp,
        dist_sync_on_step,
        sk_fn,
        preds,
        target,
        reduce,
        mdmc_reduce,
        num_classes,
        is_multiclass,
        ignore_index,
        top_k,
    ):
        if ignore_index and preds.ndim == 2:
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
        sk_fn,
        preds,
        target,
        reduce,
        mdmc_reduce,
        num_classes,
        is_multiclass,
        ignore_index,
        top_k,
    ):
        if ignore_index and preds.ndim == 2:
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

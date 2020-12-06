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
    _multilabel_multidim_prob_inputs as _mlmd_prob,
    _multilabel_multidim_inputs as _mlmd,
    _multidim_multiclass_inputs as _mdmc,
    _multidim_multiclass_prob_inputs as _mdmc_prob,
)
from tests.metrics.utils import NUM_CLASSES, THRESHOLD, EXTRA_DIM, MetricTester

torch.manual_seed(42)


def _sk_stat_scores(preds, target, reduce, num_classes, is_multiclass, ignore_index, mdmc_reduce=None):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass
    )
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if reduce != "macro" and ignore_index:
        if preds.shape[1] > 1 and 0 <= ignore_index < preds.shape[1]:
            sk_preds = np.delete(sk_preds, ignore_index, 1)
            sk_target = np.delete(sk_target, ignore_index, 1)

    if preds.shape[1] == 1 and reduce == "samples":
        sk_target = sk_target.T
        sk_preds = sk_preds.T

    samplewise = reduce == "samples" and preds.shape[1] != 1
    sk_stats = multilabel_confusion_matrix(sk_target, sk_preds, samplewise=samplewise)

    if preds.shape[1] == 1 and reduce != "samples":
        sk_stats = sk_stats[[1]].reshape(-1, 4)[:, [3, 1, 0, 2]]
    else:
        sk_stats = sk_stats.reshape(-1, 4)[:, [3, 1, 0, 2]]

    if reduce == "micro":
        sk_stats = sk_stats.sum(axis=0, keepdims=True)

    sk_stats = np.concatenate([sk_stats, sk_stats[:, [3]] + sk_stats[:, [0]]], 1)

    if reduce == "micro":
        sk_stats = sk_stats[0]

    if reduce == "macro" and ignore_index:
        if preds.shape[1] > 1 and 0 <= ignore_index < preds.shape[1]:
            sk_stats[ignore_index, :] = -1

    return sk_stats


def _sk_stat_scores_mdmc(preds, target, reduce, mdmc_reduce, num_classes, is_multiclass, ignore_index):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass
    )

    if mdmc_reduce == "global":
        shape_permute = list(range(preds.ndim))
        shape_permute[1] = shape_permute[-1]
        shape_permute[2:] = range(1, len(shape_permute) - 1)

        preds = preds.permute(*shape_permute).reshape(-1, preds.shape[1])
        target = target.permute(*shape_permute).reshape(-1, target.shape[1])

        return _sk_stat_scores(preds, target, reduce, None, False, ignore_index)
    else:  # mdmc_reduce == "samplewise"
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_stat_scores(pred_i, target_i, reduce, None, False, ignore_index)

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores)


@pytest.mark.parametrize(
    "reduce, mdmc_reduce, num_classes, inputs",
    [
        ["unknown", None, None, _binary_inputs],
        ["micro", "unknown", None, _binary_inputs],
        ["macro", None, None, _binary_inputs],
        ["micro", None, None, _mdmc_prob],
    ],
)
def test_wrong_params(reduce, mdmc_reduce, num_classes, inputs):
    with pytest.raises(ValueError):
        stat_scores(
            inputs.preds[0],
            inputs.target[0],
            reduce,
            mdmc_reduce,
            num_classes=num_classes,
        )

    with pytest.raises(ValueError):
        sts = StatScores(reduce=reduce, mdmc_reduce=mdmc_reduce, num_classes=num_classes)
        sts(inputs.preds[0], inputs.target[0])


@pytest.mark.parametrize("reduce", ["micro", "macro", "samples"])
@pytest.mark.parametrize("ignore_index", [None, 1])
@pytest.mark.parametrize(
    "preds, target, sk_fn, mdmc_reduce, num_classes, is_multiclass",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, _sk_stat_scores, None, 1, None),
        (_binary_inputs.preds, _binary_inputs.target, _sk_stat_scores, None, 1, False),
        (_ml_prob.preds, _ml_prob.target, _sk_stat_scores, None, NUM_CLASSES, None),
        (_multilabel_inputs.preds, _multilabel_inputs.target, _sk_stat_scores, None, NUM_CLASSES, False),
        (_mc_prob.preds, _mc_prob.target, _sk_stat_scores, None, NUM_CLASSES, None),
        (_multiclass_inputs.preds, _multiclass_inputs.target, _sk_stat_scores, None, NUM_CLASSES, None),
        (_mlmd_prob.preds, _mlmd_prob.target, _sk_stat_scores, None, EXTRA_DIM * NUM_CLASSES, None),
        (_mlmd.preds, _mlmd.target, _sk_stat_scores, None, EXTRA_DIM * NUM_CLASSES, False),
        (_mdmc.preds, _mdmc.target, _sk_stat_scores_mdmc, "samplewise", NUM_CLASSES, None),
        (_mdmc_prob.preds, _mdmc_prob.target, _sk_stat_scores_mdmc, "samplewise", NUM_CLASSES, None),
        (_mdmc.preds, _mdmc.target, _sk_stat_scores_mdmc, "global", NUM_CLASSES, None),
        (_mdmc_prob.preds, _mdmc_prob.target, _sk_stat_scores_mdmc, "global", NUM_CLASSES, None),
    ],
)
class TestStatScores(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
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
    ):
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
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "reduce": reduce,
                "mdmc_reduce": mdmc_reduce,
                "threshold": THRESHOLD,
                "is_multiclass": is_multiclass,
                "ignore_index": ignore_index,
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
    ):
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
            ),
            metric_args={
                "num_classes": num_classes,
                "reduce": reduce,
                "mdmc_reduce": mdmc_reduce,
                "threshold": THRESHOLD,
                "is_multiclass": is_multiclass,
                "ignore_index": ignore_index,
            },
        )

from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import multilabel_confusion_matrix

from pytorch_lightning.metrics.classification.utils import _input_format_classification
from pytorch_lightning.metrics import StatScores
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs,
    _multilabel_inputs,
    _multilabel_prob_inputs,
    _multilabel_multidim_prob_inputs,
    _multilabel_multidim_inputs,
    _multidim_multiclass_inputs,
    _multidim_multiclass_prob_inputs,
    _multidim_multiclass_prob_inputs1,
)
from tests.metrics.utils import NUM_CLASSES, THRESHOLD, EXTRA_DIM, MetricTester

torch.manual_seed(42)


def _sk_stat_scores(preds, target, reduce, num_classes, threshold, logits, is_multiclass, ignore_index, mdmc_average):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=threshold, num_classes=num_classes, logits=logits, is_multiclass=is_multiclass
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


def _sk_stat_scores_mdmc(
    preds, target, reduce, mdmc_average, num_classes, threshold, logits, is_multiclass, ignore_index
):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=threshold, num_classes=num_classes, logits=logits, is_multiclass=is_multiclass
    )

    if mdmc_average == "global":
        preds = torch.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
        target = torch.movedim(target, 1, -1).reshape(-1, target.shape[1])

        return _sk_stat_scores(preds, target, reduce, None, threshold, False, False, ignore_index, mdmc_average)
    else:  # mdmc_average == "samples"
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_stat_scores(
                pred_i, target_i, reduce, None, threshold, False, False, ignore_index, mdmc_average
            )

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores)


class StatScoresBaseTest(MetricTester):
    def test_stat_scores_mdmc(
        self,
        ddp,
        dist_sync_on_step,
        sk_fn,
        preds,
        target,
        reduce,
        mdmc_average,
        num_classes,
        logits,
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
                mdmc_average=mdmc_average,
                threshold=THRESHOLD,
                num_classes=num_classes,
                logits=logits,
                is_multiclass=is_multiclass,
                ignore_index=ignore_index,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "reduce": reduce,
                "mdmc_average": mdmc_average,
                "threshold": THRESHOLD,
                "logits": logits,
                "is_multiclass": is_multiclass,
                "ignore_index": ignore_index,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("reduce", ["micro", "macro", "samples"])
@pytest.mark.parametrize("ignore_index", [None, 1])
@pytest.mark.parametrize("mdmc_average", [None])
@pytest.mark.parametrize("sk_fn", [_sk_stat_scores])
@pytest.mark.parametrize(
    "preds, target, num_classes, logits, is_multiclass",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, 1, False, None),
        (_binary_inputs.preds, _binary_inputs.target, 1, False, False),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, NUM_CLASSES, False, None),
        (_multilabel_inputs.preds, _multilabel_inputs.target, NUM_CLASSES, False, False),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target, NUM_CLASSES, False, None),
        (_multiclass_inputs.preds, _multiclass_inputs.target, NUM_CLASSES, False, None),
        (
            _multilabel_multidim_prob_inputs.preds,
            _multilabel_multidim_prob_inputs.target,
            EXTRA_DIM * NUM_CLASSES,
            False,
            None,
        ),
        (_multilabel_multidim_inputs.preds, _multilabel_multidim_inputs.target, EXTRA_DIM * NUM_CLASSES, False, False),
    ],
)
class TestStatScores(StatScoresBaseTest):
    pass


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("reduce", ["micro", "macro", "samples"])
@pytest.mark.parametrize("ignore_index", [None, 1])
@pytest.mark.parametrize("mdmc_average", ["samplewise", "global"])
@pytest.mark.parametrize("sk_fn", [_sk_stat_scores_mdmc])
@pytest.mark.parametrize(
    "preds, target, num_classes, logits, is_multiclass",
    [
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, NUM_CLASSES, False, None),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target, NUM_CLASSES, False, None),
        (_multidim_multiclass_prob_inputs1.preds, _multidim_multiclass_prob_inputs1.target, NUM_CLASSES, False, None),
    ],
)
class TestStatScoresMDMC(StatScoresBaseTest):
    pass
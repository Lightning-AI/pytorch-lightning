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


def _sk_stat_scores(preds, target, average, num_classes, threshold, logits, is_multiclass):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=threshold, num_classes=num_classes, logits=logits, is_multiclass=is_multiclass
    )
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if preds.shape[1] == 1 and average == "samples":
        sk_target = sk_target.T
        sk_preds = sk_preds.T

    samplewise = average == "samples" and preds.shape[1] != 1
    sk_stats = multilabel_confusion_matrix(sk_target, sk_preds, samplewise=samplewise)

    if preds.shape[1] == 1 and average != "samples":
        sk_stats = sk_stats[[1]].reshape(-1, 4)[:, [3, 1, 0, 2]]
    else:
        sk_stats = sk_stats.reshape(-1, 4)[:, [3, 1, 0, 2]]

    if average == "micro":
        sk_stats = sk_stats.sum(axis=0, keepdims=True)

    sk_stats = np.concatenate([sk_stats, sk_stats[:, [3]] + sk_stats[:, [0]]], 1)

    if average == "micro":
        sk_stats = sk_stats[0]

    return sk_stats


def _sk_stat_scores_mdmc(preds, target, average, mdmc_average, num_classes, threshold, logits, is_multiclass):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=threshold, num_classes=num_classes, logits=logits, is_multiclass=is_multiclass
    )

    if mdmc_average == "global":
        preds = torch.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
        target = torch.movedim(target, 1, -1).reshape(-1, target.shape[1])

        return _sk_stat_scores(preds, target, average, None, threshold, False, False)
    else:  # mdmc_average == "samples"
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_stat_scores(pred_i, target_i, average, None, threshold, False, False)

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores)


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("average", ["micro", "macro", "samples"]) # Test just macro
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
class TestStatScores(MetricTester):
    def test_stat_scores(self, ddp, dist_sync_on_step, preds, target, num_classes, average, logits, is_multiclass):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=StatScores,
            sk_metric=partial(
                _sk_stat_scores,
                average=average,
                threshold=THRESHOLD,
                num_classes=num_classes,
                logits=logits,
                is_multiclass=is_multiclass,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "logits": logits,
                "is_multiclass": is_multiclass,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("average", ["micro", "macro", "samples"]) # Test just macro
@pytest.mark.parametrize("mdmc_average", ["global", "samples"])
@pytest.mark.parametrize(
    "preds, target, num_classes, logits, is_multiclass",
    [
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, NUM_CLASSES, False, None),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target, NUM_CLASSES, False, None),
        (_multidim_multiclass_prob_inputs1.preds, _multidim_multiclass_prob_inputs1.target, NUM_CLASSES, False, None),
    ],
)
class TestStatScoresMDMC(MetricTester):
    def test_stat_scores_mdmc(
        self, ddp, dist_sync_on_step, preds, target, average, mdmc_average, num_classes, logits, is_multiclass
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=StatScores,
            sk_metric=partial(
                _sk_stat_scores_mdmc,
                average=average,
                mdmc_average=mdmc_average,
                threshold=THRESHOLD,
                num_classes=num_classes,
                logits=logits,
                is_multiclass=is_multiclass,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "mdmc_average": mdmc_average,
                "threshold": THRESHOLD,
                "logits": logits,
                "is_multiclass": is_multiclass,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )

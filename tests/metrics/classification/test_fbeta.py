from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import fbeta_score as sk_fbeta, f1_score as sk_f1

from pytorch_lightning.metrics.classification.utils import _input_format_classification
from pytorch_lightning.metrics import F1, FBeta
from pytorch_lightning.metrics.functional import f1_score, fbeta_score, dice_score
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs as _mc_prob,
    _multidim_multiclass_inputs as _mdmc,
    _multidim_multiclass_prob_inputs as _mdmc_prob,
    _multilabel_inputs as _ml,
    _multilabel_prob_inputs as _ml_prob,
    _multilabel_multidim_prob_inputs as _mlmd_prob,
    _multilabel_multidim_inputs as _mlmd,
)
from tests.metrics.utils import EXTRA_DIM, NUM_CLASSES, THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_fbeta(preds, target, beta, num_classes, average, is_multiclass, zero_division, ignore_index, mdmc_average=None):
    if average == "none":
        average = None
    if num_classes == 1:
        average = "binary"

    labels = list(range(num_classes))
    try:
        labels.remove(ignore_index)
    except:
        pass

    sk_preds, sk_target, _ = _input_format_classification(
        preds, target, THRESHOLD, num_classes=num_classes, logits=False, is_multiclass=is_multiclass
    )
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    sk_scores = sk_fbeta(sk_target, sk_preds, beta=beta, average=average, zero_division=zero_division, labels=labels)

    if len(labels) != num_classes and not average:
        sk_scores = np.insert(sk_scores, ignore_index, np.nan)

    return sk_scores


def _sk_fbeta_mdmc(preds, target, beta, num_classes, average, is_multiclass, zero_division, ignore_index, mdmc_average):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, logits=False, is_multiclass=is_multiclass
    )

    if mdmc_average == "global":
        preds = torch.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
        target = torch.movedim(target, 1, -1).reshape(-1, target.shape[1])

        return _sk_fbeta(preds, target, beta, num_classes, average, False, zero_division, ignore_index)
    else:  # mdmc_average == "samplewise"
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_fbeta(pred_i, target_i, beta, num_classes, average, False, zero_division, ignore_index)

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores).mean()


# A single test of F1 and dice for coverage
def test_f1():
    f1 = F1(logits=False)
    score = f1(_binary_prob_inputs.preds[0], _binary_prob_inputs.target[0])
    score_fn = f1_score(_binary_prob_inputs.preds[0], _binary_prob_inputs.target[0], logits=False)
    dice = dice_score(_binary_prob_inputs.preds[0], _binary_prob_inputs.target[0], logits=False)

    sk_preds = (_binary_prob_inputs.preds[0] >= THRESHOLD).int().numpy()
    sk_target = _binary_prob_inputs.target[0].numpy()
    sk_score = sk_f1(sk_target, sk_preds)

    assert np.allclose(score.numpy(), sk_score)
    assert np.allclose(score_fn.numpy(), sk_score)
    assert np.allclose(dice.numpy(), sk_score)


######################################################################################
# Testing for MDMC inputs is partially skipped, because some cases appear where
# (with mdmc_average1 =! None, ignore_index=1, average='weighted') a sample in
# target contains only labels "1" - and as we are ignoring this index, weights of
# all labels will be zero. In this special edge case, sklearn handles the situation
# differently for each metric (recall, precision, fscore), which breaks ours handling
# everything in _reduce_scores (where the return value is 0 in this situation).
######################################################################################


@pytest.mark.parametrize("beta", [0.5, 1])
@pytest.mark.parametrize("metric_class, metric_fn", [(FBeta, fbeta_score)])
@pytest.mark.parametrize("average", ["micro", "macro", None, "weighted", "samples"])
@pytest.mark.parametrize("zero_division", [0, 1])
@pytest.mark.parametrize("ignore_index", [None, 1])
@pytest.mark.parametrize(
    "preds, target, num_classes, is_multiclass, mdmc_average, sk_wrapper",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, 1, None, None, _sk_fbeta),
        (_binary_inputs.preds, _binary_inputs.target, 1, False, None, _sk_fbeta),
        (_ml_prob.preds, _ml_prob.target, NUM_CLASSES, None, None, _sk_fbeta),
        (_ml.preds, _ml.target, NUM_CLASSES, False, None, _sk_fbeta),
        (_mc_prob.preds, _mc_prob.target, NUM_CLASSES, None, None, _sk_fbeta),
        (_multiclass_inputs.preds, _multiclass_inputs.target, NUM_CLASSES, None, None, _sk_fbeta),
        (_mlmd_prob.preds, _mlmd_prob.target, EXTRA_DIM * NUM_CLASSES, None, None, _sk_fbeta),
        (_mlmd.preds, _mlmd.target, EXTRA_DIM * NUM_CLASSES, False, None, _sk_fbeta),
        (_mdmc.preds, _mdmc.target, NUM_CLASSES, None, "global", _sk_fbeta_mdmc),
        (_mdmc_prob.preds, _mdmc_prob.target, NUM_CLASSES, None, "global", _sk_fbeta_mdmc),
        (_mdmc.preds, _mdmc.target, NUM_CLASSES, None, "samplewise", _sk_fbeta_mdmc),
        (_mdmc_prob.preds, _mdmc_prob.target, NUM_CLASSES, None, "samplewise", _sk_fbeta_mdmc),
    ],
)
class TestFBeta(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_fbeta_class(
        self,
        ddp,
        dist_sync_on_step,
        preds,
        target,
        beta,
        sk_wrapper,
        metric_class,
        metric_fn,
        is_multiclass,
        num_classes,
        average,
        mdmc_average,
        zero_division,
        ignore_index,
    ):
        if num_classes == 1 and average != "micro":
            pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

        if average == "weighted" and ignore_index is not None and mdmc_average is not None:
            pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(
                sk_wrapper,
                beta=beta,
                average=average,
                num_classes=num_classes,
                is_multiclass=is_multiclass,
                zero_division=zero_division,
                ignore_index=ignore_index,
                mdmc_average=mdmc_average,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "beta": beta,
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "logits": False,
                "is_multiclass": is_multiclass,
                "zero_division": zero_division,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )

    def test_fbeta_fn(
        self,
        preds,
        target,
        beta,
        sk_wrapper,
        metric_class,
        metric_fn,
        is_multiclass,
        num_classes,
        average,
        mdmc_average,
        zero_division,
        ignore_index,
    ):
        if num_classes == 1 and average != "micro":
            pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

        if average == "weighted" and ignore_index is not None and mdmc_average is not None:
            pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=metric_fn,
            sk_metric=partial(
                sk_wrapper,
                beta=beta,
                average=average,
                num_classes=num_classes,
                is_multiclass=is_multiclass,
                zero_division=zero_division,
                ignore_index=ignore_index,
                mdmc_average=mdmc_average,
            ),
            metric_args={
                "beta": beta,
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "logits": False,
                "is_multiclass": is_multiclass,
                "zero_division": zero_division,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
        )

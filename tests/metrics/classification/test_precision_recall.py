from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import precision_score, recall_score

from pytorch_lightning.metrics.classification.utils import _input_format_classification
from pytorch_lightning.metrics import Precision, Recall
from pytorch_lightning.metrics.functional import precision, recall
from tests.metrics.classification.inputs import (
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs as _mc_prob,
    _multidim_multiclass_inputs as _mdmc,
    _multidim_multiclass_prob_inputs as _mdmc_prob,
    _multilabel_inputs as _ml,
    _multilabel_prob_inputs as _ml_prob,
)
from tests.metrics.utils import NUM_CLASSES, THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_prec_recall(
    preds, target, sk_fn, num_classes, average, is_multiclass, zero_division, ignore_index, mdmc_average=None
):
    if average == "none":
        average = None
    if num_classes == 1:
        average = "binary"

    labels = list(range(num_classes))
    try:
        labels.remove(ignore_index)
    except ValueError:
        pass

    sk_preds, sk_target, _ = _input_format_classification(
        preds, target, THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass
    )
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    sk_scores = sk_fn(sk_target, sk_preds, average=average, zero_division=zero_division, labels=labels)

    if len(labels) != num_classes and not average:
        sk_scores = np.insert(sk_scores, ignore_index, np.nan)

    return sk_scores


def _sk_prec_recall_mdmc(
    preds, target, sk_fn, num_classes, average, is_multiclass, zero_division, ignore_index, mdmc_average
):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass
    )

    if mdmc_average == "global":
        preds = torch.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
        target = torch.movedim(target, 1, -1).reshape(-1, target.shape[1])

        return _sk_prec_recall(preds, target, sk_fn, num_classes, average, False, zero_division, ignore_index)
    else:  # mdmc_average == "samplewise"
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_prec_recall(
                pred_i, target_i, sk_fn, num_classes, average, False, zero_division, ignore_index
            )

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores).mean()


@pytest.mark.parametrize("metric, fn_metric", [(Precision, precision), (Recall, recall)])
def test_wrong_params(metric, fn_metric):
    with pytest.raises(ValueError):
        metric(zero_division=None)

    with pytest.raises(ValueError):
        fn_metric(_binary_inputs.preds[0], _binary_inputs.target[0], zero_division=None)


######################################################################################
# Testing for MDMC inputs is partially skipped, because some cases appear where
# (with mdmc_average1 =! None, ignore_index=1, average='weighted') a sample in
# target contains only labels "1" - and as we are ignoring this index, weights of
# all labels will be zero. In this special edge case, sklearn handles the situation
# differently for each metric (recall, precision, fscore), which breaks ours handling
# everything in _reduce_scores (where the return value is 0 in this situation).
######################################################################################


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes, multilabel",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, _sk_prec_recall_binary_prob, 1, False),
        (_binary_inputs.preds, _binary_inputs.target, _sk_prec_recall_binary, 1, False),
        (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target,
         _sk_prec_recall_multilabel_prob, NUM_CLASSES, True),
        (_multilabel_inputs.preds, _multilabel_inputs.target, _sk_prec_recall_multilabel, NUM_CLASSES, True),
        (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target,
         _sk_prec_recall_multiclass_prob, NUM_CLASSES, False),
        (_multiclass_inputs.preds, _multiclass_inputs.target, _sk_prec_recall_multiclass, NUM_CLASSES, False),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target,
         _sk_prec_recall_multidim_multiclass_prob, NUM_CLASSES, False),
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target,
         _sk_prec_recall_multidim_multiclass, NUM_CLASSES, False),
    ],
)
@pytest.mark.parametrize("average", ["micro", "macro", None, "weighted", "samples"])
@pytest.mark.parametrize("zero_division", [0, 1])
@pytest.mark.parametrize("ignore_index", [None, 1])
@pytest.mark.parametrize(
    "preds, target, num_classes, is_multiclass, mdmc_average, sk_wrapper",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, 1, None, None, _sk_prec_recall),
        (_binary_inputs.preds, _binary_inputs.target, 1, False, None, _sk_prec_recall),
        (_ml_prob.preds, _ml_prob.target, NUM_CLASSES, None, None, _sk_prec_recall),
        (_ml.preds, _ml.target, NUM_CLASSES, False, None, _sk_prec_recall),
        (_mc_prob.preds, _mc_prob.target, NUM_CLASSES, None, None, _sk_prec_recall),
        (_multiclass_inputs.preds, _multiclass_inputs.target, NUM_CLASSES, None, None, _sk_prec_recall),
        (_mdmc.preds, _mdmc.target, NUM_CLASSES, None, "global", _sk_prec_recall_mdmc),
        (_mdmc_prob.preds, _mdmc_prob.target, NUM_CLASSES, None, "global", _sk_prec_recall_mdmc),
        (_mdmc.preds, _mdmc.target, NUM_CLASSES, None, "samplewise", _sk_prec_recall_mdmc),
        (_mdmc_prob.preds, _mdmc_prob.target, NUM_CLASSES, None, "samplewise", _sk_prec_recall_mdmc),
    ],
)
class TestPrecisionRecall(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_precision_recall_class(
        self,
        ddp,
        dist_sync_on_step,
        preds,
        target,
        sk_wrapper,
        metric_class,
        metric_fn,
        sk_fn,
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
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                is_multiclass=is_multiclass,
                zero_division=zero_division,
                ignore_index=ignore_index,
                mdmc_average=mdmc_average,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "is_multiclass": is_multiclass,
                "zero_division": zero_division,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )

    def test_precision_recall_fn(
        self,
        preds,
        target,
        sk_wrapper,
        metric_class,
        metric_fn,
        sk_fn,
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
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                is_multiclass=is_multiclass,
                zero_division=zero_division,
                ignore_index=ignore_index,
                mdmc_average=mdmc_average,
            ),
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "is_multiclass": is_multiclass,
                "zero_division": zero_division,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
        )

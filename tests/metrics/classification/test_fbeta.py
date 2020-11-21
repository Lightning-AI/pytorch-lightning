from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import fbeta_score, f1_score

from pytorch_lightning.metrics.classification.utils import _input_format_classification
from pytorch_lightning.metrics import FBeta, F1
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
    _multidim_multiclass_prob_inputs1,
)
from tests.metrics.utils import EXTRA_DIM, NUM_CLASSES, THRESHOLD, MetricTester

torch.manual_seed(42)


def _sk_fbeta(preds, target, beta, sk_fn, num_classes, average, logits, is_multiclass, zero_division, ignore_index):
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
        preds, target, THRESHOLD, num_classes=num_classes, logits=logits, is_multiclass=is_multiclass
    )
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    sk_scores = sk_fn(sk_target, sk_preds, beta=beta, average=average, zero_division=zero_division, labels=labels)

    if len(labels) != num_classes and not average:
        sk_scores = np.insert(sk_scores, ignore_index, np.nan)

    return sk_scores


def _sk_fbeta_mdmc(
    preds, target, sk_fn, beta, num_classes, average, logits, is_multiclass, zero_division, ignore_index, mdmc_average
):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, logits=logits, is_multiclass=is_multiclass
    )

    if mdmc_average == "global":
        preds = torch.movedim(preds, 1, -1).reshape(-1, preds.shape[1])
        target = torch.movedim(target, 1, -1).reshape(-1, target.shape[1])

        return _sk_fbeta(preds, target, beta, sk_fn, num_classes, average, logits, False, zero_division, ignore_index)
    else:  # mdmc_average == "samples"
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_fbeta(
                pred_i, target_i, beta, sk_fn, num_classes, average, logits, False, zero_division, ignore_index
            )

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores).mean()


@pytest.mark.parametrize("ddp", [False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("average", ["micro"])  # Only micro, as this is equiv to "binary" in sklearn
@pytest.mark.parametrize("zero_division", [0, 1])
@pytest.mark.parametrize("ignore_index", [None])
@pytest.mark.parametrize(
    "preds, target, num_classes, logits, is_multiclass",
    [
        (_binary_prob_inputs.preds, _binary_prob_inputs.target, 1, False, None),
        (_binary_inputs.preds, _binary_inputs.target, 1, False, False),
    ],
)
@pytest.mark.parametrize("beta", [1.0, 0.5])
@pytest.mark.parametrize("metric_class, sk_fn", [(FBeta, fbeta_score)])
class TestFBetaBinary(MetricTester):
    def test_fbeta_binary(
        self,
        ddp,
        dist_sync_on_step,
        preds,
        target,
        beta,
        metric_class,
        sk_fn,
        logits,
        is_multiclass,
        num_classes,
        average,
        zero_division,
        ignore_index,
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(
                _sk_fbeta,
                beta=beta,
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                logits=logits,
                is_multiclass=is_multiclass,
                zero_division=zero_division,
                ignore_index=ignore_index,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "beta": beta,
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "logits": logits,
                "is_multiclass": is_multiclass,
                "zero_division": zero_division,
                "ignore_index": ignore_index,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )


# A single test of F1 for coverage
def test_f1():
    f1 = F1(logits=False)
    score = f1(_binary_prob_inputs.preds[0], _binary_prob_inputs.target[0])

    sk_preds = (_binary_prob_inputs.preds[0] >= THRESHOLD).int().numpy()
    sk_target = _binary_prob_inputs.target[0].numpy()
    sk_score = f1_score(sk_target, sk_preds)

    assert np.allclose(score.numpy(), sk_score)


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("average", ["micro", "macro", "none", None, "weighted", "samples"])
@pytest.mark.parametrize("zero_division", [0, 1])
@pytest.mark.parametrize("ignore_index", [None, 1])
@pytest.mark.parametrize("beta", [0.5, 1])
@pytest.mark.parametrize(
    "preds, target, num_classes, logits, is_multiclass",
    [
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
@pytest.mark.parametrize("metric_class, sk_fn", [(FBeta, fbeta_score)])
class TestFBetaNormal(MetricTester):
    def test_fbeta_normal(
        self,
        ddp,
        dist_sync_on_step,
        preds,
        target,
        beta,
        metric_class,
        sk_fn,
        logits,
        is_multiclass,
        num_classes,
        average,
        zero_division,
        ignore_index,
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(
                _sk_fbeta,
                beta=beta,
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                logits=logits,
                is_multiclass=is_multiclass,
                zero_division=zero_division,
                ignore_index=ignore_index,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "beta": beta,
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "logits": logits,
                "is_multiclass": is_multiclass,
                "zero_division": zero_division,
                "ignore_index": ignore_index,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )


# ######################################################################################
# # Testing for MDMC inputs is split into two cases, because some cases appear where
# # (with mdmc_average='samplewise', ignore_index=1, average='weighted') a sample in
# # target contains only labels "1" - and as we are ignoring this index, weights of
# # all labels will be zero. In this special edge case, sklearn handles the situation
# # differently for each metric (recall, precision, fscore), which breaks ours handling
# # everything in _reduce_scores (where the return value is 0 in this situation).
# ######################################################################################


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("average", ["micro", "macro", "none", "weighted", "samples"])
@pytest.mark.parametrize("mdmc_average", ["global", "samplewise"])
@pytest.mark.parametrize("zero_division", [0, 1])
@pytest.mark.parametrize("ignore_index", [None])
@pytest.mark.parametrize("metric_class, sk_fn", [(FBeta, fbeta_score)])
@pytest.mark.parametrize("beta", [0.5, 1])
@pytest.mark.parametrize(
    "preds, target, num_classes, logits, is_multiclass",
    [
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, NUM_CLASSES, False, None),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target, NUM_CLASSES, False, None),
        (_multidim_multiclass_prob_inputs1.preds, _multidim_multiclass_prob_inputs1.target, NUM_CLASSES, False, None),
    ],
)
class TestFBetaMDMC1(MetricTester):
    def test_fbeta_mdmc1(
        self,
        ddp,
        dist_sync_on_step,
        preds,
        target,
        beta,
        metric_class,
        sk_fn,
        logits,
        is_multiclass,
        num_classes,
        average,
        mdmc_average,
        zero_division,
        ignore_index,
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(
                _sk_fbeta_mdmc,
                beta=beta,
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                logits=logits,
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
                "logits": logits,
                "is_multiclass": is_multiclass,
                "zero_division": zero_division,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )


@pytest.mark.parametrize("ddp", [False])  # True was basically already checked, speeds up the testing
@pytest.mark.parametrize("dist_sync_on_step", [False])  # True was basically already checked, speeds up the testing
@pytest.mark.parametrize("average", ["micro", "macro", "none", "samples"])
@pytest.mark.parametrize("mdmc_average", ["samplewise", "global"])
@pytest.mark.parametrize("zero_division", [0, 1])
@pytest.mark.parametrize("ignore_index", [1])
@pytest.mark.parametrize("metric_class, sk_fn", [(FBeta, fbeta_score)])
@pytest.mark.parametrize("beta", [0.5, 1])
@pytest.mark.parametrize(
    "preds, target, num_classes, logits, is_multiclass",
    [
        (_multidim_multiclass_inputs.preds, _multidim_multiclass_inputs.target, NUM_CLASSES, False, None),
        (_multidim_multiclass_prob_inputs.preds, _multidim_multiclass_prob_inputs.target, NUM_CLASSES, False, None),
        (_multidim_multiclass_prob_inputs1.preds, _multidim_multiclass_prob_inputs1.target, NUM_CLASSES, False, None),
    ],
)
class TestFBetaMDMC2(MetricTester):
    def test_fbeta_mdmc2(
        self,
        ddp,
        dist_sync_on_step,
        preds,
        target,
        beta,
        metric_class,
        sk_fn,
        logits,
        is_multiclass,
        num_classes,
        average,
        mdmc_average,
        zero_division,
        ignore_index,
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(
                _sk_fbeta_mdmc,
                beta=beta,
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                logits=logits,
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
                "logits": logits,
                "is_multiclass": is_multiclass,
                "zero_division": zero_division,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )
import os
import pytest
import torch
import numpy as np
from collections import namedtuple

from functools import partial

from pytorch_lightning.metrics import Precision, Recall
from sklearn.metrics import precision_score, recall_score

from tests.metrics.utils import compute_batch, setup_ddp
from tests.metrics.utils import NUM_BATCHES, NUM_PROCESSES, BATCH_SIZE, NUM_CLASSES, THRESHOLD

from tests.metrics.classification.utils import (
    _binary_prob_inputs,
    _binary_inputs,
    _multilabel_prob_inputs,
    _multilabel_inputs,
    _multiclass_prob_inputs,
    _multiclass_inputs,
    _multidim_multiclass_prob_inputs,
    _multidim_multiclass_inputs,
)

torch.manual_seed(42)


def _binary_prob_sk_metric(preds, target, sk_fn=precision_score, average='micro'):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_fn(y_true=sk_target, y_pred=sk_preds, average='binary')


def _binary_sk_metric(preds, target, sk_fn=precision_score, average='micro'):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_fn(y_true=sk_target, y_pred=sk_preds, average='binary')


def _multilabel_prob_sk_metric(preds, target, sk_fn=precision_score, average='micro'):
    sk_preds = (preds.view(-1, NUM_CLASSES).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1, NUM_CLASSES).numpy()

    return sk_fn(y_true=sk_target, y_pred=sk_preds, average=average)


def _multilabel_sk_metric(preds, target, sk_fn=precision_score, average='micro'):
    sk_preds = preds.view(-1, NUM_CLASSES).numpy()
    sk_target = target.view(-1, NUM_CLASSES).numpy()

    return sk_fn(y_true=sk_target, y_pred=sk_preds, average=average)


def _multiclass_prob_sk_metric(preds, target, sk_fn=precision_score, average='micro'):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_fn(y_true=sk_target, y_pred=sk_preds, average=average)


def _multiclass_sk_metric(preds, target, sk_fn=precision_score, average='micro'):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_fn(y_true=sk_target, y_pred=sk_preds, average=average)


def _multidim_multiclass_prob_sk_metric(preds, target, sk_fn=precision_score, average='micro'):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_fn(y_true=sk_target, y_pred=sk_preds, average=average)


def _multidim_multiclass_sk_metric(preds, target, sk_fn=precision_score, average='micro'):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_fn(y_true=sk_target, y_pred=sk_preds, average=average)


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("average", ['micro', 'macro'])
@pytest.mark.parametrize("preds, target, sk_metric, num_classes, multilabel", [
    (_binary_prob_inputs.preds, _binary_prob_inputs.target, _binary_prob_sk_metric, 1, False),
    (_binary_inputs.preds, _binary_inputs.target, _binary_sk_metric, 1, False),
    (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, _multilabel_prob_sk_metric, NUM_CLASSES, True),
    (_multilabel_inputs.preds, _multilabel_inputs.target, _multilabel_sk_metric, NUM_CLASSES, True),
    (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target, _multiclass_prob_sk_metric, NUM_CLASSES, False),
    (_multiclass_inputs.preds, _multiclass_inputs.target, _multiclass_sk_metric, NUM_CLASSES, False),
    (
        _multidim_multiclass_prob_inputs.preds,
        _multidim_multiclass_prob_inputs.target,
        _multidim_multiclass_prob_sk_metric,
        NUM_CLASSES,
        False
    ),
    (
        _multidim_multiclass_inputs.preds,
        _multidim_multiclass_inputs.target,
        _multidim_multiclass_sk_metric,
        NUM_CLASSES,
        False
    )
])
@pytest.mark.parametrize(
    "metric_class, sk_fn",
    [
        (Precision, precision_score),
        (Recall, recall_score),
    ],
)
def test_precision_recall(
    ddp,
    dist_sync_on_step,
    preds,
    target,
    sk_metric,
    metric_class,
    sk_fn,
    num_classes,
    multilabel,
    average
):
    compute_batch(
        preds,
        target,
        metric_class,
        partial(sk_metric, sk_fn=sk_fn, average=average),
        dist_sync_on_step,
        ddp,
        metric_args={
            "num_classes": num_classes,
            "average": average,
            "multilabel": multilabel,
            "threshold": THRESHOLD
        },
        check_dist_sync_on_step=False if average == 'macro' else True,
        check_batch=False if average == 'macro' else True,
    )

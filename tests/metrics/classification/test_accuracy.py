import os
import pytest
import torch
import os
import numpy as np
from collections import namedtuple

from pytorch_lightning.metrics.classification.accuracy import Accuracy
from sklearn.metrics import accuracy_score

from tests.metrics.utils import compute_batch, setup_ddp
from tests.metrics.utils import NUM_BATCHES, NUM_PROCESSES, BATCH_SIZE

torch.manual_seed(42)

# global vars
num_classes = 5
threshold = 0.5
extra_dim = 3

Input = namedtuple('Input', ["preds", "target"])


def test_accuracy_invalid_shape():
    with pytest.raises(ValueError):
        acc = Accuracy()
        acc.update(preds=torch.rand(1), target=torch.rand(1, 2, 3))


_binary_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)


def _binary_prob_sk_metric(preds, target):
    sk_preds = (preds.view(-1).numpy() >= threshold).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


_binary_inputs = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE,)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE,))
)


def _binary_sk_metric(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


_multilabel_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_classes),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, num_classes))
)


def _multilabel_prob_sk_metric(preds, target):
    sk_preds = (preds.view(-1).numpy() >= threshold).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


_multilabel_inputs = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, num_classes)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, num_classes))
)


def _multilabel_sk_metric(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


_multiclass_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_classes),
    target=torch.randint(high=num_classes, size=(NUM_BATCHES, BATCH_SIZE))
)


def _multiclass_prob_sk_metric(preds, target):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


_multiclass_inputs = Input(
    preds=torch.randint(high=num_classes, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=num_classes, size=(NUM_BATCHES, BATCH_SIZE))
)


def _multiclass_sk_metric(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


_multidim_multiclass_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_classes, extra_dim),
    target=torch.randint(high=num_classes, size=(NUM_BATCHES, BATCH_SIZE, extra_dim))
)


def _multidim_multiclass_prob_sk_metric(preds, target):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


_multidim_multiclass_inputs = Input(
    preds=torch.randint(high=num_classes, size=(NUM_BATCHES, extra_dim, BATCH_SIZE)),
    target=torch.randint(high=num_classes, size=(NUM_BATCHES, extra_dim, BATCH_SIZE))
)


def _multidim_multiclass_sk_metric(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return accuracy_score(y_true=sk_target, y_pred=sk_preds)


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("ddp_sync_on_step", [True, False])
@pytest.mark.parametrize("preds, target, sk_metric", [
    (_binary_prob_inputs.preds, _binary_prob_inputs.target, _binary_prob_sk_metric),
    (_binary_inputs.preds, _binary_inputs.target, _binary_sk_metric),
    (_multilabel_prob_inputs.preds, _multilabel_prob_inputs.target, _multilabel_prob_sk_metric),
    (_multilabel_inputs.preds, _multilabel_inputs.target, _multilabel_sk_metric),
    (_multiclass_prob_inputs.preds, _multiclass_prob_inputs.target, _multiclass_prob_sk_metric),
    (_multiclass_inputs.preds, _multiclass_inputs.target, _multiclass_sk_metric),
    (
        _multidim_multiclass_prob_inputs.preds,
        _multidim_multiclass_prob_inputs.target,
        _multidim_multiclass_prob_sk_metric
    ),
    (
        _multidim_multiclass_inputs.preds,
        _multidim_multiclass_inputs.target,
        _multidim_multiclass_sk_metric
    )
])
def test_accuracy(ddp, ddp_sync_on_step, preds, target, sk_metric):
    compute_batch(preds, target, Accuracy, sk_metric, ddp_sync_on_step, ddp, metric_args={"threshold": threshold})

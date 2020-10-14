import os
import pytest
import numpy as np
import torch

from collections import namedtuple
from tests.metrics.utils import (
    NUM_BATCHES,
    NUM_PROCESSES,
    BATCH_SIZE,
    NUM_CLASSES,
    EXTRA_DIM,
    THRESHOLD
)

Input = namedtuple('Input', ["preds", "target"])


_binary_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)


_binary_inputs = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE,)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE,))
)


_multilabel_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
)


_multilabel_inputs = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
)


_multiclass_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)


_multiclass_inputs = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)


_multidim_multiclass_prob_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM))
)


_multidim_multiclass_inputs = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, EXTRA_DIM, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, EXTRA_DIM, BATCH_SIZE))
)

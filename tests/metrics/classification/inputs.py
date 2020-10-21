from collections import namedtuple

import torch

from tests.metrics.utils import (
    NUM_BATCHES,
    BATCH_SIZE,
    NUM_CLASSES,
    EXTRA_DIM
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

# Generate edge multilabel edge case, where nothing matches (scores are undefined)
__temp_preds = torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
__temp_target = abs(__temp_preds - 1)

_multilabel_inputs_no_match = Input(
    preds=__temp_preds,
    target=__temp_target
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

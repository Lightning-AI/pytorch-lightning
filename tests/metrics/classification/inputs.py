from collections import namedtuple

import torch

from tests.metrics.utils import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, NUM_CLASSES

Input = namedtuple('Input', ["preds", "target"])

_input_binary_prob = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_binary = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multilabel_prob = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
)

_input_multilabel_multidim_prob = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM))
)

_input_multilabel = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
)

_input_multilabel_multidim = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM))
)

# Generate edge multilabel edge case, where nothing matches (scores are undefined)
__temp_preds = torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
__temp_target = abs(__temp_preds - 1)

_input_multilabel_no_match = Input(preds=__temp_preds, target=__temp_target)

__mc_prob_preds = torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
__mc_prob_preds = __mc_prob_preds / __mc_prob_preds.sum(dim=2, keepdim=True)

_input_multiclass_prob = Input(
    preds=__mc_prob_preds, target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multiclass = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)

__mdmc_prob_preds = torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
__mdmc_prob_preds = __mdmc_prob_preds / __mdmc_prob_preds.sum(dim=2, keepdim=True)

_input_multidim_multiclass_prob = Input(
    preds=__mdmc_prob_preds, target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM))
)

_input_multidim_multiclass = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM))
)

import pytest
import sys
from collections import namedtuple
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

import tests.base.develop_utils as tutils
from pytorch_lightning.metrics import (
    Accuracy,
    ConfusionMatrix,
    #PrecisionRecallCurve
    Precision,
    Recall,
    AveragePrecision,
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    average_precision_score

)

# example structure
Example = namedtuple('example', ['name', 'lightning_metric', 'comparing_metric', 'test_input'])

# setup some standard testcases
multiclass_example = [(torch.randint(10, (100,)), torch.randint(10, (100,)))]
binary_example = [(torch.randint(2, (100,)), torch.randint(2, (100,)))]
multiclass_and_binary_example = [*multiclass_example, *binary_example]
binary_example_logits = (torch.randint(2, (100,)), torch.randint(5, (100,)))


EXAMPLES = [
    Example('accuracy',
            Accuracy,
            accuracy_score,
            multiclass_and_binary_example),
    Example('confusion_matrix_without_normalize',
            ConfusionMatrix,
            confusion_matrix,
            multiclass_and_binary_example),
    Example('confusion_matrix_with_normalize',
            partial(ConfusionMatrix, normalize=True),
            partial(confusion_matrix, normalize='true'),
            multiclass_and_binary_example),
    Example('precision_score',
            Precision,
            partial(precision_score, average='micro'),
            multiclass_and_binary_example),
    Example('recall_score',
            Precision,
            partial(precision_score, average='micro'),
            multiclass_and_binary_example),
    Example('average_precision',
            AveragePrecision,
            average_precision_score,
            binary_example)

]

def idsfn(val):
    """ Return id for current example being tested """
    return val.name

def _setup_ddp(rank, worldsize):
    """ setup ddp enviroment for test """
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def _test_fn(rank, worldsize, example):
    _setup_ddp(rank, worldsize)

    lightning_metric = example.lightning_metric()
    comparing_metric = example.comparing_metric

    for test_input in example.test_input:
        lightning_val = lightning_metric(*test_input)
        comparing_val = comparing_metric(*test_input.reverse())
        assert lightning_val == comparing_val


@pytest.mark.skipif(sys.platform == "win32" , reason="DDP not available on windows")
@pytest.mark.parametrize("example", EXAMPLES, ids=idsfn)
def test_ddp(example):
    """Make sure that metrics are correctly sync and reduced in DDP mode"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_test_fn, args=(worldsize, example), )


@pytest.mark.parametrize("example", EXAMPLES, ids=idsfn)
def test_multi_batch(example):
    """ test that aggregation works for multiple batches """
    lightning_metric = example.lightning_metric()
    comparing_metric = example.comparing_metric

    for test_input in example.test_input:
        for i in range(2): # for lightning device in 2 artificially batches
            _ = lightning_metric(*[ti[i::2] for ti in test_input])

        lightning_val = lightning_metric.aggregated
        comparing_val = comparing_metric(*[ti.numpy() for ti in reversed(test_input)])

        assert np.allclose(lightning_val.numpy(), comparing_val, rtol=1e-3)



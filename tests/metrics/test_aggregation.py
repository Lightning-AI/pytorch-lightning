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
    # PrecisionRecallCurve
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
N_samples = 200
multiclass_example = [(torch.randint(10, (N_samples,)), torch.randint(10, (N_samples,)))]
binary_example = [(torch.randint(2, (N_samples,)), torch.randint(2, (N_samples,)))]
multiclass_and_binary_example = [*multiclass_example, *binary_example]
binary_example_logits = (torch.randint(2, (N_samples,)), torch.randint(5, (N_samples,)))


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
    # Example('average_precision',
    #         AveragePrecision,
    #         average_precision_score,
    #         binary_example)

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


def _test_ddp_single_batch(rank, worldsize, lightning_metric, comparing_metric, test_inputs):
    _setup_ddp(rank, worldsize)

    # Setup metric for ddp
    lightning_metric = lightning_metric()
    for test_input in test_inputs:
        lightning_val = lightning_metric(*[ti[rank::2] for ti in test_input])
        comparing_val = comparing_metric(*[ti.numpy() for ti in reversed(test_input)])
        assert np.allclose(lightning_val.numpy(), comparing_val, rtol=1e-3)


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize("example", EXAMPLES, ids=idsfn)
def test_ddp(example):
    """Make sure that metrics are correctly sync and reduced in DDP mode"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_test_ddp_single_batch,
             args=(worldsize,
                   example.lightning_metric,
                   example.comparing_metric,
                   example.test_input),
             nprocs=worldsize)


@pytest.mark.parametrize("example", EXAMPLES, ids=idsfn)
def test_multi_batch(example):
    """ test that aggregation works for multiple batches """
    lightning_metric = example.lightning_metric()
    comparing_metric = example.comparing_metric

    for test_input in example.test_input:
        for i in range(2):  # for lightning device in 2 artificially batches
            _ = lightning_metric(*[ti[i::2] for ti in test_input])
        lightning_val = lightning_metric.aggregated
        comparing_val = comparing_metric(*[ti.numpy() for ti in reversed(test_input)])
        assert np.allclose(lightning_val.numpy(), comparing_val, rtol=1e-3)


def _test_ddp_multi_batch(rank, worldsize, lightning_metric, comparing_metric, test_inputs):
    _setup_ddp(rank, worldsize)

    # Setup metric for ddp
    lightning_metric = lightning_metric()
    for test_input in test_inputs:
        for i in range(2):  # artificially divide samples between batches and processes
            _ = lightning_metric(*[ti[i + worldsize * rank::4] for ti in test_input])
        lightning_val = lightning_metric.aggregated
        comparing_val = comparing_metric(*[ti.numpy() for ti in reversed(test_input)])
        assert np.allclose(lightning_val.numpy(), comparing_val, rtol=1e-3)


@pytest.mark.parametrize("example", EXAMPLES, ids=idsfn)
def test_ddp_multi_batch(example):
    """ test that aggregation works fine with in DDP mode and multiple batches """
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_test_ddp_multi_batch,
             args=(worldsize,
                   example.lightning_metric,
                   example.comparing_metric,
                   example.test_input),
             nprocs=worldsize)

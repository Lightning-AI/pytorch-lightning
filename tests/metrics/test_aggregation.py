import pytest
import sys
from collections import namedtuple
from functools import partial
import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

from tests.base import EvalModelTemplate
from pytorch_lightning import Trainer
import tests.base.develop_utils as tutils
from pytorch_lightning.metrics import (
    Accuracy,
    ConfusionMatrix,
    PrecisionRecallCurve,
    Precision,
    Recall,
    AveragePrecision,
    AUROC,
    FBeta,
    F1,
    ROC,
    MulticlassROC,
    MulticlassPrecisionRecallCurve,
    DiceCoefficient,
    IoU,
    MAE,
    MSE,
    RMSE,
    RMSLE,
    PSNR,
    SSIM,
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    fbeta_score,
    f1_score,
    roc_curve,
    jaccard_score,
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error
)

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# example structure
TestCase = namedtuple('example', ['name', 'lightning_metric', 'comparing_metric', 'test_input'])

# setup some standard testcases
NB_SAMPLES = 200
multiclass_example = [(torch.randint(10, (NB_SAMPLES,)), torch.randint(10, (NB_SAMPLES,)))]
binary_example = [(torch.randint(2, (NB_SAMPLES,)), torch.randint(2, (NB_SAMPLES,)))]
multiclass_and_binary_example = [*multiclass_example, *binary_example]
binary_example_logits = (torch.randint(2, (NB_SAMPLES,)), torch.randint(5, (NB_SAMPLES,)))
multiclass_example_probs = (torch.randint(10, (NB_SAMPLES,)), torch.randn((NB_SAMPLES, 10)).softmax(-1))
regression_example = [(torch.rand((NB_SAMPLES,)), torch.rand((NB_SAMPLES,)))]


# construct additional test functions
def root_mean_squared_error(x, y):
    return math.sqrt(mean_squared_error(x, y))


def root_mean_squared_log_error(x, y):
    return math.sqrt(mean_squared_log_error(x, y))


# Define testcases
# TODO: update remaining metrics and uncomment the corresponding test cases
TESTS = [
    TestCase('accuracy',
             Accuracy,
             accuracy_score,
             multiclass_and_binary_example),
    TestCase('confusion matrix without normalize',
             ConfusionMatrix,
             confusion_matrix,
             multiclass_and_binary_example),
    TestCase('confusion matrix with normalize',
             partial(ConfusionMatrix, normalize=True),
             partial(confusion_matrix, normalize='true'),
             multiclass_and_binary_example),
    # TestCase('precision recall curve',
    #          PrecisionRecallCurve,
    #          precision_recall_curve,
    #          binary_example),
    TestCase('precision',
             Precision,
             partial(precision_score, average='micro'),
             multiclass_and_binary_example),
    TestCase('recall',
             Recall,
             partial(recall_score, average='micro'),
             multiclass_and_binary_example),
    # TestCase('average_precision',
    #          AveragePrecision,
    #          average_precision_score,
    #          binary_example),
    # TestCase('auroc',
    #          AUROC,
    #          roc_auc_score,
    #          binary_example),
    TestCase('f beta',
             partial(FBeta, beta=2),
             partial(fbeta_score, average='micro', beta=2),
             multiclass_and_binary_example),
    TestCase('f1',
             F1,
             partial(f1_score, average='micro'),
             multiclass_and_binary_example),
    # TestCase('roc',
    #          ROC,
    #          roc_curve,
    #          binary_example),
    # TestCase('multiclass roc',
    #          MulticlassROC,
    #          multiclass_roc,
    #          binary_example),
    # TestCase('multiclass precision recall curve',
    #          MulticlassPrecisionRecallCurve,
    #          multiclass_precision_recall_curve,
    #          binary_example),
    # TestCase('dice coefficient',
    #          DiceCoefficient,
    #          partial(f1_score, average='micro'),
    #          multiclass_and_binary_example),
    # TestCase('intersection over union',
    #          IoU,
    #          partial(jaccard_score, average='macro'),
    #          binary_example),
    TestCase('mean squared error',
             MSE,
             mean_squared_error,
             regression_example),
    TestCase('root mean squared error',
             RMSE,
             root_mean_squared_error,
             regression_example),
    TestCase('mean absolute error',
             MAE,
             mean_absolute_error,
             regression_example),
    TestCase('root mean squared log error',
             RMSLE,
             root_mean_squared_log_error,
             regression_example),
    TestCase('peak signal-to-noise ratio',
             partial(PSNR, data_range=10),
             partial(peak_signal_noise_ratio, data_range=10),
             regression_example),
    # TestCase('structual similarity index measure',
    #          SSIM,
    #          structural_similarity,
    #          regression_example)
]


# Utility test functions
def _idsfn(test):
    """ Return id for current example being tested """
    return test.name


def _setup_ddp(rank, worldsize):
    """ setup ddp enviroment for testing """
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def comparing_fn(lightning_val, comparing_val, rtol=1e-03, atol=1e-08):
    """ function for comparing output, both multi and single output"""
    # multi output
    if isinstance(comparing_val, tuple):
        for l_score, c_score in zip(lightning_val, comparing_val):
            assert np.allclose(l_score.numpy(), c_score, rtol, atol)
    else:  # single output
        assert np.allclose(lightning_val.numpy(), comparing_val, rtol, atol)


# ===== Tests start here =====
def _test_ddp_single_batch(rank, worldsize, lightning_metric, comparing_metric, test_inputs):
    """ ddp testing function, divide test_inputs equally between all processes """
    _setup_ddp(rank, worldsize)

    # Setup metric for ddp
    lightning_metric = lightning_metric()
    for test_input in test_inputs:
        # rank 0 receives sample 0,2,4,...
        # rank 1 receives sample 1,3,5,...
        lightning_val = lightning_metric(*[ti[rank::2] for ti in test_input])

        comparing_val = comparing_metric(*[ti.numpy() for ti in reversed(test_input)])

        comparing_fn(lightning_val, comparing_val)


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize("test", TESTS, ids=_idsfn)
def test_ddp(test):
    """Make sure that metrics are correctly sync and reduced in DDP mode"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_test_ddp_single_batch,
             args=(worldsize,
                   test.lightning_metric,
                   test.comparing_metric,
                   test.test_input),
             nprocs=worldsize)


@pytest.mark.parametrize("test", TESTS, ids=_idsfn)
def test_multi_batch(test):
    """ test that aggregation works for multiple batches """
    lightning_metric = test.lightning_metric()
    comparing_metric = test.comparing_metric

    for test_input in test.test_input:
        for i in range(2):  # for lightning device in 2 artificially batches
            # first batch consist of samples 0,2,4,...
            # second batch consist of samples 1,3,5,...
            _ = lightning_metric(*[ti[i::2] for ti in test_input])
        lightning_val = lightning_metric.aggregated
        comparing_val = comparing_metric(*[ti.numpy() for ti in reversed(test_input)])

        comparing_fn(lightning_val, comparing_val)


@pytest.mark.parametrize("test", TESTS, ids=_idsfn)
def test_multi_batch_unequal_sizes(test):
    """ test that aggregation works for multiple batches with uneven sizes """
    lightning_metric = test.lightning_metric()
    comparing_metric = test.comparing_metric

    for test_input in test.test_input:
        for i in range(2):  # for lightning device in 2 artificially batches
            if i == 0:  # allocate 3/4 of data to the first batch
                _ = lightning_metric(*[ti[:int(3 / 4 * len(ti))] for ti in test_input])
            else:
                _ = lightning_metric(*[ti[int(3 / 4 * len(ti)):] for ti in test_input])
        lightning_val = lightning_metric.aggregated
        comparing_val = comparing_metric(*[ti.numpy() for ti in reversed(test_input)])

        comparing_fn(lightning_val, comparing_val)


def _test_ddp_multi_batch(rank, worldsize, lightning_metric, comparing_metric, test_inputs):
    """ ddp testing function, test that metric works with aggregation over multiple
        devices and multiple batches """
    _setup_ddp(rank, worldsize)

    # Setup metric for ddp
    lightning_metric = lightning_metric()
    for test_input in test_inputs:
        for i in range(2):  # artificially divide samples between batches and processes
            # rank 0, batch 0 consist of samples 0,4,8,...
            # rank 0, batch 1 consist of samples 1,5,9,...
            # rank 1, batch 0 consist of samples 2,6,10,...
            # rank 1, batch 1 consist of samples 3,7,11,...
            _ = lightning_metric(*[ti[i + worldsize * rank::4] for ti in test_input])
        lightning_val = lightning_metric.aggregated
        comparing_val = comparing_metric(*[ti.numpy() for ti in reversed(test_input)])

        comparing_fn(lightning_val, comparing_val)


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize("test", TESTS, ids=_idsfn)
def test_ddp_multi_batch(test):
    """ test that aggregation works fine with in DDP mode and multiple batches """
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_test_ddp_multi_batch,
             args=(worldsize,
                   test.lightning_metric,
                   test.comparing_metric,
                   test.test_input),
             nprocs=worldsize)

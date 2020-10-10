import os
import sys
import pytest
import pickle
from typing import Callable

import torch
import numpy as np

from pytorch_lightning.metrics import Metric

NUM_PROCESSES = 2
NUM_BATCHES = 10
BATCH_SIZE = 16
NUM_CLASSES = 5
EXTRA_DIM = 3
THRESHOLD = 0.5


def setup_ddp(rank, world_size):
    """ Setup ddp enviroment """
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ['MASTER_PORT'] = '8088'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _compute_batch(
    rank: int,
    preds: torch.Tensor,
    target: torch.Tensor,
    metric_class: Metric,
    sk_metric: Callable,
    ddp_sync_on_step: bool,
    worldsize: int = 1,
    metric_args: dict = {},
    check_dist_sync_on_step: bool = True,
    check_batch: bool = True,
):
    """ Utility function doing the actual comparison between lightning metric
        and reference metric
    """
    # Instanciate lightning metric
    metric = metric_class(compute_on_step=True, ddp_sync_on_step=ddp_sync_on_step, **metric_args)

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)

    # Only use ddp if world size
    if worldsize > 1:
        setup_ddp(rank, worldsize)

    for i in range(rank, NUM_BATCHES, worldsize):
        batch_result = metric(preds[i], target[i])

        if metric.dist_sync_on_step:
            if rank == 0:
                ddp_preds = torch.stack([preds[i + r] for r in range(worldsize)])
                ddp_target = torch.stack([target[i + r] for r in range(worldsize)])
                sk_batch_result = sk_metric(ddp_preds, ddp_target)
                # assert for dist_sync_on_step
                if check_dist_sync_on_step:
                    assert np.allclose(batch_result.numpy(), sk_batch_result)
        else:
            sk_batch_result = sk_metric(preds[i], target[i])
            # assert for batch
            if check_batch:
                assert np.allclose(batch_result.numpy(), sk_batch_result)

    # check on all batches on all ranks
    result = metric.compute()
    assert isinstance(result, torch.Tensor)

    total_preds = torch.stack([preds[i] for i in range(NUM_BATCHES)])
    total_target = torch.stack([target[i] for i in range(NUM_BATCHES)])
    sk_result = sk_metric(total_preds, total_target)

    # assert after aggregation
    assert np.allclose(result.numpy(), sk_result)


def compute_batch(
    preds: torch.Tensor,
    target: torch.Tensor,
    metric_class: Metric,
    sk_metric: Callable,
    ddp_sync_on_step: bool,
    ddp: bool = False,
    metric_args: dict = {},
    check_dist_sync_on_step: bool = True,
    check_batch: bool = True,
):
    """ Utility function for comparing the result between a lightning class
        metric and another metric (often sklearns)

        Args:
            preds: prediction tensor
            target: target tensor
            metric_class: lightning metric class to test
            sk_metric: function to compare with
            ddp_sync_on_step: bool, determine if values should be reduce on step
            ddp: bool, determine if test should run in ddp mode
            metric_args: dict, additional kwargs that are use when instanciating
                the lightning metric
            check_dist_sync_on_step: assert for dist_sync_on_step
            check_batch: assert for each batch
    """
    if ddp:
        if sys.platform == "win32":
            pytest.skip("DDP not supported on windows")

        torch.multiprocessing.spawn(
            _compute_batch, args=(
                preds,
                target,
                metric_class,
                sk_metric,
                dist_sync_on_step,
                NUM_PROCESSES,
                metric_args,
                check_dist_sync_on_step,
                check_batch,
            ),
            nprocs=NUM_PROCESSES
        )
    else:
        # first args: rank, last args: world size
        _compute_batch(
            rank=0,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=sk_metric,
            dist_sync_on_step=dist_sync_on_step,
            worldsize=1,
            metric_args=metric_args,
            check_dist_sync_on_step=check_dist_sync_on_step,
            check_batch=check_batch,
        )

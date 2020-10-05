import torch
import numpy as np
import os
import sys
import pytest

NUM_PROCESSES = 2
NUM_BATCHES = 10
BATCH_SIZE = 16


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ['MASTER_PORT'] = '8088'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _compute_batch(rank, preds, target, metric_class, sk_metric, ddp_sync_on_step, worldsize=1, metric_args={}):

    metric = metric_class(compute_on_step=True, ddp_sync_on_step=ddp_sync_on_step, **metric_args)

    # Only use ddp if world size
    if worldsize > 1:
        setup_ddp(rank, worldsize)

    for i in range(rank, NUM_BATCHES, worldsize):
        batch_result = metric(preds[i], target[i])

        if metric.ddp_sync_on_step:
            if rank == 0:
                ddp_preds = torch.stack([preds[i + r] for r in range(worldsize)])
                ddp_target = torch.stack([target[i + r] for r in range(worldsize)])
                sk_batch_result = sk_metric(ddp_preds, ddp_target)
                assert np.allclose(batch_result.numpy(), sk_batch_result)
        else:
            sk_batch_result = sk_metric(preds[i], target[i])
            assert np.allclose(batch_result.numpy(), sk_batch_result)

    # check on all batches on all ranks
    result = metric.compute()
    assert isinstance(result, torch.Tensor)

    total_preds = torch.stack([preds[i] for i in range(NUM_BATCHES)])
    total_target = torch.stack([target[i] for i in range(NUM_BATCHES)])
    sk_result = sk_metric(total_preds, total_target)

    assert np.allclose(result.numpy(), sk_result)


def compute_batch(preds, target, metric_class, sk_metric, ddp_sync_on_step, ddp=False, metric_args={}):
    if ddp:
        if sys.platform == "win32":
            pytest.skip("DDP not supported on windows")

        torch.multiprocessing.spawn(
            _compute_batch, args=(preds, target, metric_class, sk_metric, ddp_sync_on_step, NUM_PROCESSES, metric_args),
            nprocs=NUM_PROCESSES
        )
    else:
        # first args: rank, last args: world size
        _compute_batch(0, preds, target, metric_class, sk_metric, ddp_sync_on_step, 1, metric_args)

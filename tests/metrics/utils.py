import os
import pickle
import sys
from functools import partial
from typing import Callable

import numpy as np
import pytest
import torch
from torch.multiprocessing import Pool, set_start_method

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

    if torch.distributed.is_available():
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _compute_batch(
    rank: int,
    worldsize: int,
    preds: torch.Tensor,
    target: torch.Tensor,
    metric_class: Metric,
    sk_metric: Callable,
    dist_sync_on_step: bool,
    metric_args: dict = {},
    check_dist_sync_on_step: bool = True,
    check_batch: bool = True,
):
    """ Utility function doing the actual comparison between lightning metric
        and reference metric.

        Args:
            rank: rank of current process
            worldsize: number of processes
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_class: lightning metric class that should be tested
            sk_metric: callable function that is used for comparison
            dist_sync_on_step: bool, if true will synchronize metric state across
                processes at each ``forward()``
            metric_args: dict with additional arguments used for class initialization
            check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                calculated per batch per device (and not just at the end)
            check_batch: bool, if true will check if the metric is also correctly
                calculated across devices for each batch (and not just at the end)
    """
    # Instanciate lightning metric
    metric = metric_class(compute_on_step=True, dist_sync_on_step=dist_sync_on_step, **metric_args)

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)

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


class MetricTester:
    """ Class used for efficiently run alot of parametrized tests in ddp mode.
        Makes sure that ddp is only setup once and that pool of processes are
        used for all tests.

        All tests should subclass from this and implement a new method called
            `test_metric_name`
        where the method `self.run_metric_test` is called inside.
    """

    def setup_class(self):
        """ Setup the metric class. This will spawn the pool of workers that are
            used for metric testing and setup_ddp
        """
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        self.poolSize = NUM_PROCESSES
        self.pool = Pool(processes=self.poolSize)
        self.pool.starmap(setup_ddp, [(rank, self.poolSize) for rank in range(self.poolSize)])

    def teardown_class(self):
        """ Close pool of workers """
        self.pool.close()
        self.pool.join()

    def run_metric_test(
        self,
        ddp: bool,
        preds: torch.Tensor,
        target: torch.Tensor,
        metric_class: Metric,
        sk_metric: Callable,
        dist_sync_on_step: bool,
        metric_args: dict = {},
        check_dist_sync_on_step: bool = True,
        check_batch: bool = True,
    ):
        """ Main method that should be used for testing. Call this inside testing
            methods.

            Args:
                ddp: bool, if running in ddp mode or not
                preds: torch tensor with predictions
                target: torch tensor with targets
                metric_class: lightning metric class that should be tested
                sk_metric: callable function that is used for comparison
                dist_sync_on_step: bool, if true will synchronize metric state across
                    processes at each ``forward()``
                metric_args: dict with additional arguments used for class initialization
                check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                    calculated per batch per device (and not just at the end)
                check_batch: bool, if true will check if the metric is also correctly
                    calculated across devices for each batch (and not just at the end)
        """
        if ddp:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")

            self.pool.starmap(
                partial(
                    _compute_batch,
                    preds=preds,
                    target=target,
                    metric_class=metric_class,
                    sk_metric=sk_metric,
                    dist_sync_on_step=dist_sync_on_step,
                    metric_args=metric_args,
                    check_dist_sync_on_step=check_dist_sync_on_step,
                    check_batch=check_batch,
                ),
                [(rank, self.poolSize) for rank in range(self.poolSize)],
            )
        else:
            _compute_batch(
                0,
                1,
                preds=preds,
                target=target,
                metric_class=metric_class,
                sk_metric=sk_metric,
                dist_sync_on_step=dist_sync_on_step,
                metric_args=metric_args,
                check_dist_sync_on_step=check_dist_sync_on_step,
                check_batch=check_batch,
            )

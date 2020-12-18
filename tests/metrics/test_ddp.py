import sys

import pytest
import torch

from pytorch_lightning.metrics import Metric
from tests.metrics.test_metric import Dummy
from tests.metrics.utils import setup_ddp

torch.manual_seed(42)


def _test_ddp_sum(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": torch.sum}
    dummy.foo = torch.tensor(1)

    dummy._sync_dist()
    assert dummy.foo == worldsize


def _test_ddp_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": torch.cat}
    dummy.foo = [torch.tensor([1])]
    dummy._sync_dist()
    assert torch.all(torch.eq(dummy.foo, torch.tensor([1, 1])))


def _test_ddp_sum_cat(rank, worldsize):
    setup_ddp(rank, worldsize)
    dummy = Dummy()
    dummy._reductions = {"foo": torch.cat, "bar": torch.sum}
    dummy.foo = [torch.tensor([1])]
    dummy.bar = torch.tensor(1)
    dummy._sync_dist()
    assert torch.all(torch.eq(dummy.foo, torch.tensor([1, 1])))
    assert dummy.bar == worldsize


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize("process", [_test_ddp_cat, _test_ddp_sum, _test_ddp_sum_cat])
def test_ddp(process):
    torch.multiprocessing.spawn(process, args=(2,), nprocs=2)


def _test_non_contiguous_tensors(rank, worldsize):
    setup_ddp(rank, worldsize)

    class DummyMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("x", default=[], dist_reduce_fx=None)

        def update(self, x):
            self.x.append(x)

        def compute(self):
            x = torch.cat(self.x, dim=0)
            return x.sum()

    metric = DummyMetric()
    metric.update(torch.randn(10, 5)[:, 0])


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_non_contiguous_tensors():
    """ Test that gather_all operation works for non contiguous tensors """
    torch.multiprocessing.spawn(_test_non_contiguous_tensors, args=(2,), nprocs=2)


def _test_running_accumulation(rank, worldsize):
    class RunningMetric(Metric):
        def __init__(self):
            super().__init__(auto_reset_on_compute=False)
            self.add_state("x", torch.tensor(0.0), dist_reduce_fx="sum")

        def update(self, x):
            self.x += x

        def compute(self):
            return self.x

    running_metric = RunningMetric()

    for _ in range(2):
        _ = running_metric(1.0)
    acc_val = running_metric.compute()

    assert acc_val == 4.0, "wrong accumulation of metric states"
    if rank == 0:
        assert running_metric.x != torch.tensor(0.)
    else:
        assert running_metric.x == torch.tensor(0.)

    for _ in range(2):
        val = running_metric(1.0)
    acc_val = running_metric.compute()

    assert acc_val == 8.0, "wrong accumulation of metric states"
    if rank == 0:
        assert running_metric.x != torch.tensor(0.)
    else:
        assert running_metric.x == torch.tensor(0.)

    running_metric.reset()
    assert running_metric.x == torch.tensor(0.), "metric state should have been reset"


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_running_accumulation():
    """ Test that disabling automatic reset on compute works in ddp"""
    torch.multiprocessing.spawn(_test_running_accumulation, args=(2,), nprocs=2)

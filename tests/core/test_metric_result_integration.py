# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics import Metric

import tests.helpers.utils as tutils
from pytorch_lightning.core.step_result import Result
from tests.helpers.runif import RunIf


class DummyMetric(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("x", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x):
        self.x += x

    def compute(self):
        return self.x


def _setup_ddp(rank, worldsize):
    import os

    os.environ["MASTER_ADDR"] = "localhost"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def _ddp_test_fn(rank, worldsize):
    _setup_ddp(rank, worldsize)
    torch.tensor([1.0])

    metric_a = DummyMetric()
    metric_b = DummyMetric()
    metric_c = DummyMetric()

    # dist_sync_on_step is False by default
    result = Result()

    for epoch in range(3):
        cumulative_sum = 0

        for i in range(5):
            metric_a(i)
            metric_b(i)
            metric_c(i)

            cumulative_sum += i

            result.log('a', metric_a, on_step=True, on_epoch=True)
            result.log('b', metric_b, on_step=False, on_epoch=True)
            result.log('c', metric_c, on_step=True, on_epoch=False)

            batch_log = result.get_batch_log_metrics()
            batch_expected = {"a_step": i, "a": i, "c": i}
            assert set(batch_log.keys()) == set(batch_expected.keys())
            for k in batch_expected.keys():
                assert batch_expected[k] == batch_log[k]

        epoch_log = result.get_epoch_log_metrics()
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults['x']
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']

        epoch_expected = {"b": cumulative_sum * worldsize, "a_epoch": cumulative_sum * worldsize}

        assert set(epoch_log.keys()) == set(epoch_expected.keys())
        for k in epoch_expected.keys():
            assert epoch_expected[k] == epoch_log[k]


@RunIf(skip_windows=True)
def test_result_reduce_ddp():
    """Make sure result logging works with DDP"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_ddp_test_fn, args=(worldsize, ), nprocs=worldsize)


def test_result_metric_integration():
    metric_a = DummyMetric()
    metric_b = DummyMetric()
    metric_c = DummyMetric()

    result = Result()

    for epoch in range(3):
        cumulative_sum = 0

        for i in range(5):
            metric_a(i)
            metric_b(i)
            metric_c(i)

            cumulative_sum += i

            result.log('a', metric_a, on_step=True, on_epoch=True)
            result.log('b', metric_b, on_step=False, on_epoch=True)
            result.log('c', metric_c, on_step=True, on_epoch=False)

            batch_log = result.get_batch_log_metrics()
            batch_expected = {"a_step": i, "a": i, "c": i}
            assert set(batch_log.keys()) == set(batch_expected.keys())
            for k in batch_expected.keys():
                assert batch_expected[k] == batch_log[k]

        epoch_log = result.get_epoch_log_metrics()
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults['x']
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']

        epoch_expected = {"b": cumulative_sum, "a_epoch": cumulative_sum}

        assert set(epoch_log.keys()) == set(epoch_expected.keys())
        for k in epoch_expected.keys():
            assert epoch_expected[k] == epoch_log[k]

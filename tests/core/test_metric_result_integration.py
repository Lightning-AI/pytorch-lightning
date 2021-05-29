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
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics import Metric

import tests.helpers.utils as tutils
from pytorch_lightning.core.result import DefaultMetricsKeys, ResultCollection
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

    metric_a = metric_a.to(f"cuda:{rank}")
    metric_b = metric_b.to(f"cuda:{rank}")
    metric_c = metric_c.to(f"cuda:{rank}")

    # dist_sync_on_step is False by default
    result = ResultCollection(True, torch.device(f"cuda:{rank}"))

    for _ in range(3):
        cumulative_sum = 0

        result.on_epoch_end_reached = False

        for i in range(5):

            result.batch_idx = i

            metric_a(i)
            metric_b(i)
            metric_c(i)

            cumulative_sum += i

            result.log('h', 'a', metric_a, on_step=True, on_epoch=True, lightning_attribute_name="metric_a")
            result.log('h', 'b', metric_b, on_step=False, on_epoch=True, lightning_attribute_name="metric_b")
            result.log('h', 'c', metric_c, on_step=True, on_epoch=False, lightning_attribute_name="metric_c")

            batch_log = result.get_batch_metrics()[DefaultMetricsKeys.LOG]
            batch_expected = {"a_step": i, "c": i}
            assert set(batch_log.keys()) == set(batch_expected.keys())
            for k in batch_expected.keys():
                assert batch_expected[k] == batch_log[k]

        result.on_epoch_end_reached = True

        epoch_log = result.get_epoch_metrics()[DefaultMetricsKeys.LOG]
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults['x'], (metric_a.x, metric_a._defaults['x'])
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']

        epoch_expected = {"b": cumulative_sum * worldsize, "a_epoch": cumulative_sum * worldsize}

        assert set(epoch_log.keys()) == set(epoch_expected.keys())
        for k in epoch_expected.keys():
            assert epoch_expected[k] == epoch_log[k]


@RunIf(skip_windows=True, min_gpus=2)
def test_result_reduce_ddp():
    """Make sure result logging works with DDP"""
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_ddp_test_fn, args=(worldsize, ), nprocs=worldsize)


def test_result_metric_integration():
    metric_a = DummyMetric()
    metric_b = DummyMetric()
    metric_c = DummyMetric()

    result = ResultCollection(True, torch.device("cpu"))

    for _ in range(3):
        cumulative_sum = 0

        result.on_epoch_end_reached = False

        for i in range(5):

            # need to set batch_idx
            result.batch_idx = i

            metric_a(i)
            metric_b(i)
            metric_c(i)

            cumulative_sum += i

            result.log('h', 'a', metric_a, on_step=True, on_epoch=True, lightning_attribute_name="metric_a")
            result.log('h', 'b', metric_b, on_step=False, on_epoch=True, lightning_attribute_name="metric_b")
            result.log('h', 'c', metric_c, on_step=True, on_epoch=False, lightning_attribute_name="metric_c")

            batch_log = result.get_batch_metrics()[DefaultMetricsKeys.LOG]
            batch_expected = {"a_step": i, "c": i}
            assert set(batch_log.keys()) == set(batch_expected.keys())
            for k in batch_expected.keys():
                assert batch_expected[k] == batch_log[k]

        result.on_epoch_end_reached = True

        epoch_log = result.get_epoch_metrics()[DefaultMetricsKeys.LOG]
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults['x']
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']

        epoch_expected = {"b": cumulative_sum, "a_epoch": cumulative_sum}

        assert set(epoch_log.keys()) == set(epoch_expected.keys())
        for k in epoch_expected.keys():
            assert epoch_expected[k] == epoch_log[k]


def test_result_collection_restoration():

    _result = None
    metric_a = DummyMetric()
    metric_b = DummyMetric()
    metric_c = DummyMetric()

    result = ResultCollection(True, torch.device("cpu"))

    for _ in range(2):

        result.on_epoch_end_reached = False
        cumulative_sum = 0

        for i in range(3):

            result.batch_idx = i

            a = metric_a(i)
            b = metric_b(i)
            c = metric_c(i)

            cumulative_sum += i

            result.log('training_step', 'a', metric_a, on_step=True, on_epoch=True, lightning_attribute_name="metric_a")
            result.log(
                'training_step', 'b', metric_b, on_step=False, on_epoch=True, lightning_attribute_name="metric_b"
            )
            result.log(
                'training_step', 'c', metric_c, on_step=True, on_epoch=False, lightning_attribute_name="metric_c"
            )
            result.log('training_step', 'a_1', a, on_step=True, on_epoch=True)
            result.log('training_step', 'b_1', b, on_step=False, on_epoch=True)
            result.log('training_step', 'c_1', [c, c], on_step=True, on_epoch=False)

            batch_log = result.metrics[DefaultMetricsKeys.LOG]
            batch_expected = {"a_step": i, "c": i, "a_1_step": i, "c_1": [i, i]}

            assert set(batch_log.keys()) == set(batch_expected.keys())
            for k in batch_expected.keys():
                assert batch_expected[k] == batch_log[k]

            _result = deepcopy(result)
            state_dict = result.state_dict()

            result = ResultCollection(True, torch.device("cpu"))
            result.load_from_state_dict(
                state_dict, {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "metric_c": metric_c,
                    "metric_a_end": metric_a
                }
            )

            assert _result.items() == result.items()

        result.on_epoch_end_reached = True
        _result.on_epoch_end_reached = True

        epoch_log = result.metrics[DefaultMetricsKeys.LOG]
        _epoch_log = _result.metrics[DefaultMetricsKeys.LOG]

        assert epoch_log == _epoch_log

        epoch_expected = {'a_1_epoch', 'a_epoch', 'b', 'b_1'}

        assert set(epoch_log.keys()) == epoch_expected, epoch_log.keys()
        for k in list(epoch_expected):
            if k in {'a_epoch', 'b'}:
                assert epoch_log[k] == cumulative_sum
            else:
                assert epoch_log[k] == 1

        result.log(
            'train_epoch_end', 'a', metric_a, on_step=False, on_epoch=True, lightning_attribute_name="metric_a_end"
        )

        _result.reset()
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults['x'], (metric_a.x, metric_a._defaults['x'])
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']


def test_result_collection_simple_loop():

    result = ResultCollection(True, torch.device("cpu"))

    result.log('a0', 'a', torch.tensor(0.), on_step=True, on_epoch=True)
    result.log('a1', 'a', torch.tensor(0.), on_step=True, on_epoch=True)

    for epoch in range(2):

        result.on_epoch_end_reached = False

        result.log('b0', 'a', torch.tensor(1.) + epoch, on_step=True, on_epoch=True)
        result.log('b1', 'a', torch.tensor(1.) + epoch, on_step=True, on_epoch=True)

        for batch_idx, batch_size in enumerate(range(2)):

            result.batch_idx = batch_idx

            result.log('c0', 'a', torch.tensor(2.) + epoch, on_step=True, on_epoch=True)
            result.log('c1', 'a', torch.tensor(2.) + epoch, on_step=True, on_epoch=True)
            result.log('c2', 'a', torch.tensor(2.) + epoch, on_step=True, on_epoch=True)

        result.on_epoch_end_reached = True

        result.log('d0', 'a', torch.tensor(3.) + epoch, on_step=False, on_epoch=True)
        result.log('d1', 'a', torch.tensor(3.) + epoch, on_step=False, on_epoch=True)

        assert result['a0.a'].value == torch.tensor(0.)
        assert result['a0.a'].cumulated_batch_size == torch.tensor(1.)
        assert result['a1.a'].value == torch.tensor(0.)
        assert result['a1.a'].cumulated_batch_size == torch.tensor(1.)

        assert result['b0.a'].value == torch.tensor(1.) + epoch
        assert result['b0.a'].cumulated_batch_size == torch.tensor(1.)
        assert result['b1.a'].value == torch.tensor(1.) + epoch
        assert result['b1.a'].cumulated_batch_size == torch.tensor(1.)

        assert result['c0.a'].value == torch.tensor(4.) + epoch * (batch_size + 1)
        assert result['c0.a'].cumulated_batch_size == torch.tensor(2.)
        assert result['c1.a'].value == torch.tensor(4.) + epoch * (batch_size + 1)
        assert result['c1.a'].cumulated_batch_size == torch.tensor(2.)
        assert result['c2.a'].value == torch.tensor(4.) + epoch * (batch_size + 1)
        assert result['c2.a'].cumulated_batch_size == torch.tensor(2.)

        assert result['d0.a'].value == torch.tensor(3.) + epoch
        assert result['d0.a'].cumulated_batch_size == torch.tensor(1.)
        assert result['d1.a'].value == torch.tensor(3.) + epoch
        assert result['d1.a'].cumulated_batch_size == torch.tensor(1.)

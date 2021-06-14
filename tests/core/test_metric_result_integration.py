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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.connectors.logger_connector.result import MetricSource, ResultCollection
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class DummyMetric(Metric):

    def __init__(self, name: str = None):
        super().__init__()
        self.add_state("x", torch.tensor(0), dist_reduce_fx="sum")
        self.name = name

    def update(self, x):
        self.x += x

    def compute(self):
        return self.x

    def extra_repr(self) -> str:
        return str(self.name) if self.name else ''


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

    result = ResultCollection(True, torch.device(f"cuda:{rank}"))

    for _ in range(3):
        cumulative_sum = 0
        for i in range(5):
            metric_a(i)
            metric_b(i)
            metric_c(i)

            cumulative_sum += i

            result.log('h', 'a', metric_a, on_step=True, on_epoch=True)
            result.log('h', 'b', metric_b, on_step=False, on_epoch=True)
            result.log('h', 'c', metric_c, on_step=True, on_epoch=False)

            batch_log = result.metrics(True)[MetricSource.LOG]
            assert batch_log == {"a_step": i, "c": i}

        epoch_log = result.metrics(False)[MetricSource.LOG]
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults['x'], (metric_a.x, metric_a._defaults['x'])
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']

        assert epoch_log == {"b": cumulative_sum * worldsize, "a_epoch": cumulative_sum * worldsize}


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
        for i in range(5):
            metric_a(i)
            metric_b(i)
            metric_c(i)

            cumulative_sum += i

            result.log('h', 'a', metric_a, on_step=True, on_epoch=True)
            result.log('h', 'b', metric_b, on_step=False, on_epoch=True)
            result.log('h', 'c', metric_c, on_step=True, on_epoch=False)

            batch_log = result.metrics(True)[MetricSource.LOG]
            assert batch_log == {"a_step": i, "c": i}

        epoch_log = result.metrics(False)[MetricSource.LOG]
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults['x']
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']

        assert epoch_log == {"b": cumulative_sum, "a_epoch": cumulative_sum}

    assert str(result) == (
        "ResultCollection(True, cpu, {"
        "'h.a': ResultMetric(value=DummyMetric()), "
        "'h.b': ResultMetric(value=DummyMetric()), "
        "'h.c': ResultMetric(value=DummyMetric())"
        "})"
    )


def test_result_collection_simple_loop():
    result = ResultCollection(True, torch.device("cpu"))
    current_fx_name = None
    batch_idx = None

    def lightning_log(fx, *args, **kwargs):
        nonlocal current_fx_name
        if current_fx_name != fx and batch_idx in (None, 0):
            result.reset(metrics=False, fx=fx)
        result.log(fx, *args, **kwargs)
        current_fx_name = fx

    lightning_log('a0', 'a', torch.tensor(0.), on_step=True, on_epoch=True)
    lightning_log('a1', 'a', torch.tensor(0.), on_step=True, on_epoch=True)
    for epoch in range(2):
        lightning_log('b0', 'a', torch.tensor(1.) + epoch, on_step=True, on_epoch=True)
        lightning_log('b1', 'a', torch.tensor(1.) + epoch, on_step=True, on_epoch=True)
        for batch_idx in range(2):
            lightning_log('c0', 'a', torch.tensor(2.) + epoch, on_step=True, on_epoch=True)
            lightning_log('c1', 'a', torch.tensor(2.) + epoch, on_step=True, on_epoch=True)
            lightning_log('c2', 'a', torch.tensor(2.) + epoch, on_step=True, on_epoch=True)
        batch_idx = None
        lightning_log('d0', 'a', torch.tensor(3.) + epoch, on_step=False, on_epoch=True)
        lightning_log('d1', 'a', torch.tensor(3.) + epoch, on_step=False, on_epoch=True)

        for k in ('a0.a', 'a1.a'):
            assert result[k].value == torch.tensor(0.), k
            assert result[k].cumulated_batch_size == torch.tensor(1.), k

        for k in ('b0.a', 'b1.a'):
            assert result[k].value == torch.tensor(1.) + epoch, k
            assert result[k].cumulated_batch_size == torch.tensor(1.), k

        for k in ('c0.a', 'c1.a', 'c2.a'):
            assert result[k].value == torch.tensor(4.) + epoch * 2, k
            assert result[k].cumulated_batch_size == torch.tensor(2.), k

        for k in ('d0.a', 'd1.a'):
            assert result[k].value == torch.tensor(3.) + epoch, k
            assert result[k].cumulated_batch_size == torch.tensor(1.), k


def test_result_collection_restoration():
    """"
    This test make sure metrics are properly reloaded on failure.
    """

    result = ResultCollection(True, torch.device("cpu"))
    _result = None
    metric_a = DummyMetric('a')
    metric_b = DummyMetric('b')
    metric_c = DummyMetric('c')
    metric_d = DummyMetric('d')
    current_fx_name = None
    batch_idx = None

    def lightning_log(fx, *args, **kwargs):
        nonlocal current_fx_name
        if current_fx_name != fx and batch_idx in (None, 0):
            result.reset(metrics=False, fx=fx)
        result.log(fx, *args, **kwargs)
        current_fx_name = fx

    for _ in range(2):

        cumulative_sum = 0

        for i in range(3):

            a = metric_a(i)
            b = metric_b(i)
            c = metric_c(i)
            metric_d(i)

            cumulative_sum += i

            metric = metric_a if i < 1 else metric_d
            lightning_log('training_step', 'a', metric, on_step=True, on_epoch=True)
            lightning_log('training_step', 'b', metric_b, on_step=False, on_epoch=True)
            lightning_log('training_step', 'c', metric_c, on_step=True, on_epoch=False)
            lightning_log('training_step', 'a_1', a, on_step=True, on_epoch=True)
            lightning_log('training_step', 'b_1', b, on_step=False, on_epoch=True)
            lightning_log('training_step', 'c_1', {'1': c, '2': c}, on_step=True, on_epoch=False)

            batch_log = result.metrics(on_step=True)[MetricSource.LOG]
            assert set(batch_log) == {"a_step", "c", "a_1_step", "c_1"}
            assert set(batch_log['c_1']) == {'1', '2'}
            _result = deepcopy(result)
            state_dict = result.state_dict()

            result = ResultCollection(True, torch.device("cpu"))
            result.load_from_state_dict(state_dict)

            assert _result.items() == result.items()
            assert _result["training_step.c_1"].meta == result["training_step.c_1"].meta

        batch_idx = None

        epoch_log = result.metrics(on_step=False)[MetricSource.LOG]
        _epoch_log = result.metrics(on_step=False)[MetricSource.LOG]
        assert epoch_log == _epoch_log

        assert set(epoch_log) == {'a_1_epoch', 'a_epoch', 'b', 'b_1'}
        for k in epoch_log:
            if k in {'a_epoch', 'b'}:
                assert epoch_log[k] == cumulative_sum
            else:
                assert epoch_log[k] == 1

        lightning_log('train_epoch_end', 'a', metric_a, on_step=False, on_epoch=True)

        result.reset()
        _result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults['x']
        assert metric_b.x == metric_b._defaults['x']
        assert metric_c.x == metric_c._defaults['x']


def test_lightning_module_logging_result_collection(tmpdir):

    class LoggingModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.metric = DummyMetric()

        def training_step(self, batch, batch_idx):
            v = self.metric(batch_idx)
            self.log_dict({"v": v, "m": self.metric})
            return super().training_step(batch, batch_idx)

        def on_save_checkpoint(self, checkpoint) -> None:
            state_dict = self.trainer.train_loop.results.state_dict()
            checkpoint["result_collections"] = state_dict
            self.trainer.train_loop.results.load_from_state_dict(state_dict)
            assert self.trainer.train_loop.results['training_step.v'].meta.sync.fn is None
            return super().on_save_checkpoint(checkpoint)

    model = LoggingModel()
    ckpt = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=3, limit_train_batches=2, limit_val_batches=2, callbacks=[ckpt]
    )
    trainer.fit(model)

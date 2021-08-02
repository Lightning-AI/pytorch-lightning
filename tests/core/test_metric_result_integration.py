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
import pickle
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics import Metric

import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.connectors.logger_connector.result import _Sync, MetricSource, ResultCollection
from tests.helpers import BoringModel
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

    result = ResultCollection(True, torch.device(f"cuda:{rank}"))

    for _ in range(3):
        cumulative_sum = 0
        for i in range(5):
            metric_a(i)
            metric_b(i)
            metric_c(i)

            cumulative_sum += i

            result.log("h", "a", metric_a, on_step=True, on_epoch=True)
            result.log("h", "b", metric_b, on_step=False, on_epoch=True)
            result.log("h", "c", metric_c, on_step=True, on_epoch=False)

            batch_log = result.metrics(True)[MetricSource.LOG]
            assert batch_log == {"a_step": i, "c": i}

        epoch_log = result.metrics(False)[MetricSource.LOG]
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults["x"], (metric_a.x, metric_a._defaults["x"])
        assert metric_b.x == metric_b._defaults["x"]
        assert metric_c.x == metric_c._defaults["x"]

        assert epoch_log == {"b": cumulative_sum * worldsize, "a_epoch": cumulative_sum * worldsize}


@RunIf(skip_windows=True, min_gpus=2)
def test_result_reduce_ddp():
    """Make sure result logging works with DDP"""
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(_ddp_test_fn, args=(worldsize,), nprocs=worldsize)


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

            result.log("h", "a", metric_a, on_step=True, on_epoch=True)
            result.log("h", "b", metric_b, on_step=False, on_epoch=True)
            result.log("h", "c", metric_c, on_step=True, on_epoch=False)

            batch_log = result.metrics(True)[MetricSource.LOG]
            assert batch_log == {"a_step": i, "c": i}

        epoch_log = result.metrics(False)[MetricSource.LOG]
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults["x"]
        assert metric_b.x == metric_b._defaults["x"]
        assert metric_c.x == metric_c._defaults["x"]

        assert epoch_log == {"b": cumulative_sum, "a_epoch": cumulative_sum}

    result.minimize = torch.tensor(1.0)
    result.extra = {}
    assert str(result) == (
        "ResultCollection("
        "minimize=1.0, "
        "{"
        "'h.a': ResultMetric('a', value=DummyMetric()), "
        "'h.b': ResultMetric('b', value=DummyMetric()), "
        "'h.c': ResultMetric('c', value=DummyMetric())"
        "})"
    )
    assert repr(result) == (
        "{"
        "True, "
        "device(type='cpu'), "
        "minimize=tensor(1.), "
        "{'h.a': ResultMetric('a', value=DummyMetric()), "
        "'h.b': ResultMetric('b', value=DummyMetric()), "
        "'h.c': ResultMetric('c', value=DummyMetric()), "
        "'_extra': {}}"
        "}"
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

    lightning_log("a0", "a", torch.tensor(0.0), on_step=True, on_epoch=True)
    lightning_log("a1", "a", torch.tensor(0.0), on_step=True, on_epoch=True)
    for epoch in range(2):
        lightning_log("b0", "a", torch.tensor(1.0) + epoch, on_step=True, on_epoch=True)
        lightning_log("b1", "a", torch.tensor(1.0) + epoch, on_step=True, on_epoch=True)
        for batch_idx in range(2):
            lightning_log("c0", "a", torch.tensor(2.0) + epoch, on_step=True, on_epoch=True)
            lightning_log("c1", "a", torch.tensor(2.0) + epoch, on_step=True, on_epoch=True)
            lightning_log("c2", "a", torch.tensor(2.0) + epoch, on_step=True, on_epoch=True)
        batch_idx = None
        lightning_log("d0", "a", torch.tensor(3.0) + epoch, on_step=False, on_epoch=True)
        lightning_log("d1", "a", torch.tensor(3.0) + epoch, on_step=False, on_epoch=True)

        for k in ("a0.a", "a1.a"):
            assert result[k].value == torch.tensor(0.0), k
            assert result[k].cumulated_batch_size == torch.tensor(1.0), k

        for k in ("b0.a", "b1.a"):
            assert result[k].value == torch.tensor(1.0) + epoch, k
            assert result[k].cumulated_batch_size == torch.tensor(1.0), k

        for k in ("c0.a", "c1.a", "c2.a"):
            assert result[k].value == torch.tensor(4.0) + epoch * 2, k
            assert result[k].cumulated_batch_size == torch.tensor(2.0), k

        for k in ("d0.a", "d1.a"):
            assert result[k].value == torch.tensor(3.0) + epoch, k
            assert result[k].cumulated_batch_size == torch.tensor(1.0), k


def my_sync_dist(x, *_, **__):
    return x


def test_result_collection_restoration(tmpdir):
    """
    This test make sure metrics are properly reloaded on failure.
    """

    result = ResultCollection(True, torch.device("cpu"))
    metric_a = DummyMetric()
    metric_b = DummyMetric()
    metric_c = DummyMetric()
    metric_d = DummyMetric()
    current_fx_name = None
    batch_idx = None

    def lightning_log(fx, *args, **kwargs):
        nonlocal current_fx_name
        if current_fx_name != fx and batch_idx in (None, 0):
            result.reset(metrics=False, fx=fx)
        result.log(fx, *args, **kwargs, sync_dist_fn=my_sync_dist)
        current_fx_name = fx

    for epoch in range(2):

        cumulative_sum = 0

        for i in range(3):

            a = metric_a(i)
            b = metric_b(i)
            c = metric_c(i)
            metric_d(i)

            cumulative_sum += i

            metric = metric_a if i < 1 else metric_d
            lightning_log("training_step", "a", metric, on_step=True, on_epoch=True, metric_attribute="metric")
            lightning_log("training_step", "b", metric_b, on_step=False, on_epoch=True, metric_attribute="metric_b")
            lightning_log("training_step", "c", metric_c, on_step=True, on_epoch=False, metric_attribute="metric_c")
            lightning_log("training_step", "a_1", a, on_step=True, on_epoch=True)
            lightning_log("training_step", "b_1", b, on_step=False, on_epoch=True)
            lightning_log("training_step", "c_1", {"1": c, "2": c}, on_step=True, on_epoch=False)

            batch_log = result.metrics(on_step=True)[MetricSource.LOG]
            assert set(batch_log) == {"a_step", "c", "a_1_step", "c_1"}
            assert set(batch_log["c_1"]) == {"1", "2"}

            result_copy = deepcopy(result)
            new_result = ResultCollection(True, torch.device("cpu"))
            state_dict = result.state_dict()
            # check the sync fn was dropped
            assert "fn" not in state_dict["items"]["training_step.a"]["meta"]["_sync"]

            assert not new_result.result_metrics
            assert len(result.result_metrics) == 7 + epoch > 0

            new_result.load_state_dict(
                state_dict, metrics={"metric": metric, "metric_b": metric_b, "metric_c": metric_c}
            )
            # should match
            assert result_copy == new_result
            # the sync fn has been kept
            assert result_copy["training_step.a"].meta.sync.fn == new_result["training_step.a"].meta.sync.fn

        epoch_log = result.metrics(on_step=False)[MetricSource.LOG]
        epoch_log_copy = result_copy.metrics(on_step=False)[MetricSource.LOG]
        assert epoch_log == epoch_log_copy

        lightning_log("train_epoch_end", "a", metric_a, on_step=False, on_epoch=True)
        epoch_log = result.metrics(on_step=False)[MetricSource.LOG]
        assert epoch_log == {
            "a_1_epoch": 1,
            "a_epoch": cumulative_sum,
            "a": cumulative_sum,
            "b": cumulative_sum,
            "b_1": 1,
        }

        # make sure can be pickled
        pickle.loads(pickle.dumps(result))
        # make sure can be torch.loaded
        filepath = str(tmpdir / "result")
        torch.save(result, filepath)
        torch.load(filepath)

        # assert metric state reset to default values
        result.reset()
        assert metric_a.x == metric_a._defaults["x"]
        assert metric_b.x == metric_b._defaults["x"]
        assert metric_c.x == metric_c._defaults["x"]

        batch_idx = None


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=RunIf(min_gpus=1))))
def test_lightning_module_logging_result_collection(tmpdir, device):
    class LoggingModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metric = DummyMetric()

        def validation_step(self, batch, batch_idx):
            v = self.metric(batch_idx)
            self.log_dict({"v": v, "m": self.metric})
            return super().validation_step(batch, batch_idx)

        def on_save_checkpoint(self, checkpoint) -> None:
            results = self.trainer._results
            # simplify logic
            state_dict = results.state_dict(drop_value=False)

            # check device
            assert results["validation_step.v"].value.device.type == device
            assert state_dict["items"]["validation_step.v"]["value"].device.type == device

            # sync fn should be kept
            assert results["validation_step.v"].meta.sync.fn == self.trainer.training_type_plugin.reduce

            # sync fn dropped from the state dict
            assert "fn" not in state_dict["items"]["validation_step.v"]["meta"]["_sync"]
            results.load_state_dict(state_dict)

            # check device after loading
            assert results["validation_step.v"].value.device.type == device

            # sync fn was preserved in the original result
            assert results["validation_step.v"].meta.sync.fn == self.trainer.training_type_plugin.reduce

            # default sync fn
            new_results = ResultCollection(False, device)
            new_results.load_state_dict(state_dict, map_location="cpu")
            assert new_results["validation_step.v"].meta.sync.fn == _Sync.no_op

            # check map location
            assert new_results["validation_step.v"].value.device.type == "cpu"

    model = LoggingModel()
    ckpt = ModelCheckpoint(dirpath=tmpdir, save_on_train_epoch_end=False)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[ckpt],
        gpus=1 if device == "cuda" else 0,
    )
    trainer.fit(model)


def test_result_collection_extra_reference():
    """Unit-test to check that the `extra` dict reference is properly set."""
    rc = ResultCollection(True)
    assert rc.extra is rc["_extra"]

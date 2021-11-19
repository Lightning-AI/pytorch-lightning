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
import os
import pickle
from contextlib import suppress
from copy import deepcopy
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import ModuleDict, ModuleList
from torchmetrics import Metric, MetricCollection

import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.connectors.logger_connector.result import (
    _Metadata,
    _Sync,
    ResultCollection,
    ResultMetric,
)
from pytorch_lightning.utilities.imports import _fault_tolerant_training, _TORCH_GREATER_EQUAL_1_7
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

            batch_log = result.metrics(True)["log"]
            assert batch_log == {"a_step": i, "c": i}

        epoch_log = result.metrics(False)["log"]
        result.reset()

        # assert metric state reset to default values
        assert metric_a.x == metric_a._defaults["x"], (metric_a.x, metric_a._defaults["x"])
        assert metric_b.x == metric_b._defaults["x"]
        assert metric_c.x == metric_c._defaults["x"]

        assert epoch_log == {"b": cumulative_sum * worldsize, "a_epoch": cumulative_sum * worldsize}


@RunIf(skip_windows=True, min_gpus=2)
def test_result_reduce_ddp():
    """Make sure result logging works with DDP."""
    tutils.set_random_main_port()

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

            batch_log = result.metrics(True)["log"]
            assert batch_log == {"a_step": i, "c": i}

        epoch_log = result.metrics(False)["log"]
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
        "{'h.a': ResultMetric('a', value=DummyMetric()), "
        "'h.b': ResultMetric('b', value=DummyMetric()), "
        "'h.c': ResultMetric('c', value=DummyMetric())"
        "}}"
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
    """This test make sure metrics are properly reloaded on failure."""

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

            batch_log = result.metrics(on_step=True)["log"]
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

        epoch_log = result.metrics(on_step=False)["log"]
        epoch_log_copy = result_copy.metrics(on_step=False)["log"]
        assert epoch_log == epoch_log_copy

        lightning_log("train_epoch_end", "a", metric_a, on_step=False, on_epoch=True)
        epoch_log = result.metrics(on_step=False)["log"]
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
            assert new_results["validation_step.v"].meta.sync.fn is None

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


class DummyMeanMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum", torch.tensor(0), dist_reduce_fx=torch.sum)
        self.add_state("count", torch.tensor(0), dist_reduce_fx=torch.sum)

    def update(self, increment):
        self.sum += increment
        self.count += 1

    def compute(self):
        return self.sum // self.count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sum={self.sum}, count={self.count})"


def result_collection_reload(**kwargs):

    """This test is going to validate ResultCollection is properly being reload and final accumulation with Fault
    Tolerant Training is correct."""

    if not _fault_tolerant_training():
        pytest.skip("Fault tolerant not available")

    num_processes = kwargs.get("gpus", 1)

    class CustomException(Exception):
        pass

    class ExtendedBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.breaking_batch_idx = 3
            self.has_validated_sum = False
            self.dummy_metric = DummyMeanMetric()

        @property
        def results(self):
            return self.trainer.fit_loop._results

        def training_step(self, batch, batch_idx):

            # In the training step, we will accumulate metrics using batch_idx from 0 to 4
            # Without failure, we would expect to get `total=10 * world_size` and `num_batches=5 * world_size`
            # Therefore, compute on `epoch_end` should provide 2 as `10 / 5`.
            # However, below we will simulate a failure on `batch_idx=3`.

            if self.trainer.fit_loop.restarting:
                self.log("tracking", batch_idx, on_step=True, on_epoch=True)
                self.log("tracking_2", batch_idx, on_step=True, on_epoch=True, sync_dist=True)

                self.dummy_metric(batch_idx)
                self.log("tracking_metric", self.dummy_metric, on_step=True, on_epoch=True)

                value = self.results["training_step.tracking_metric"].value
                value_2 = self.results["training_step.tracking"].value

                # On failure, the Metric states are being accumulated on rank 0 and zeroed-out on other ranks.
                # The shift indicates we failed while the state was `shift=sign(is_global_zero > 0) * [0..3]`
                shift = 0
                if num_processes == 2:
                    shift = 3 if self.trainer.is_global_zero else -3
                expected = sum(range(batch_idx + 1)) + shift
                assert expected == value == value_2
            else:
                if batch_idx == self.breaking_batch_idx:
                    # simulate failure mid epoch
                    raise CustomException

                self.log("tracking", batch_idx, on_step=True, on_epoch=True)
                self.log("tracking_2", batch_idx, on_step=True, on_epoch=True, sync_dist=True)

                self.dummy_metric(batch_idx)
                self.log("tracking_metric", self.dummy_metric, on_step=True, on_epoch=True)

                value = self.results["training_step.tracking"].value
                assert value == sum(range(batch_idx + 1))

                value = self.results["training_step.tracking_2"]
                assert value == sum(range(batch_idx + 1))

            return super().training_step(batch, batch_idx)

        def on_epoch_end(self) -> None:
            if self.trainer.fit_loop.restarting:
                total = sum(range(5)) * num_processes
                metrics = self.results.metrics(on_step=False)
                assert self.results["training_step.tracking"].value == total
                assert metrics["callback"]["tracking"] == self.dummy_metric.compute() == 2
                assert self.results["training_step.tracking_2"].value == total
                assert metrics["callback"]["tracking_2"] == self.dummy_metric.compute() == 2
                self.has_validated_sum = True

    model = ExtendedBoringModel()
    trainer_kwargs = {"max_epochs": 1, "limit_train_batches": 5, "limit_val_batches": 0}
    trainer_kwargs.update(kwargs)
    trainer = Trainer(**trainer_kwargs)

    with suppress(CustomException):
        trainer.fit(model)
    assert not model.has_validated_sum

    tmpdir = (
        trainer.training_type_plugin.broadcast(trainer_kwargs["default_root_dir"], 0)
        if num_processes >= 2
        else trainer_kwargs["default_root_dir"]
    )
    ckpt_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, ckpt_path=ckpt_path)
    assert model.has_validated_sum


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_7, reason="Requires at least PyTorch 1.7")
def test_result_collection_reload(tmpdir):
    result_collection_reload(default_root_dir=tmpdir)


@RunIf(min_gpus=1)
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_7, reason="Requires at least PyTorch 1.7")
def test_result_collection_reload_1_gpu_ddp(tmpdir):
    result_collection_reload(default_root_dir=tmpdir, strategy="ddp", gpus=1)


@RunIf(min_gpus=2, special=True)
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_7, reason="Requires at least PyTorch 1.7")
def test_result_collection_reload_2_gpus(tmpdir):
    result_collection_reload(default_root_dir=tmpdir, strategy="ddp", gpus=2)


def test_metric_collections(tmpdir):
    """This test ensures the metric attribute is properly found even with complex nested metric structure."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.metrics_list = ModuleList([DummyMetric() for _ in range(2)])
            self.metrics_dict = ModuleDict({"a": DummyMetric(), "b": DummyMetric()})
            self.metrics_collection_dict = MetricCollection({"a": DummyMetric(), "b": DummyMetric()})
            self.metrics_collection_dict_nested = ModuleDict(
                {"a": ModuleList([ModuleDict({"b": DummyMetric()}), DummyMetric()])}
            )

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            self.metrics_list[0](batch_idx)
            self.metrics_list[1](batch_idx)

            self.metrics_dict["a"](batch_idx)
            self.metrics_dict["b"](batch_idx)

            self.metrics_collection_dict["a"](batch_idx)
            self.metrics_collection_dict["b"](batch_idx)

            self.metrics_collection_dict_nested["a"][0]["b"](batch_idx)
            self.metrics_collection_dict_nested["a"][1](batch_idx)

            self.log("a", self.metrics_list[0])
            self.log("b", self.metrics_list[1])

            self.log("c", self.metrics_dict["a"])
            self.log("d", self.metrics_dict["b"])

            self.log("e", self.metrics_collection_dict["a"])
            self.log("f", self.metrics_collection_dict["b"])

            self.log("g", self.metrics_collection_dict_nested["a"][0]["b"])
            self.log("h", self.metrics_collection_dict_nested["a"][1])

            return loss

        def on_train_epoch_end(self) -> None:
            results = self.trainer.fit_loop.epoch_loop._results
            assert results["training_step.a"].meta.metric_attribute == "metrics_list.0"
            assert results["training_step.b"].meta.metric_attribute == "metrics_list.1"

            assert results["training_step.c"].meta.metric_attribute == "metrics_dict.a"
            assert results["training_step.d"].meta.metric_attribute == "metrics_dict.b"

            assert results["training_step.e"].meta.metric_attribute == "metrics_collection_dict.a"
            assert results["training_step.f"].meta.metric_attribute == "metrics_collection_dict.b"

            assert results["training_step.g"].meta.metric_attribute == "metrics_collection_dict_nested.a.0.b"
            assert results["training_step.h"].meta.metric_attribute == "metrics_collection_dict_nested.a.1"

    model = TestModel()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_train_batches=2, limit_val_batches=0)
    trainer.fit(model)


def test_metric_result_computed_check():
    """Unittest ``_get_cache`` with multielement tensors."""
    metadata = _Metadata("foo", "bar", on_epoch=True, enable_graph=True)
    metadata.sync = _Sync()
    rm = ResultMetric(metadata, is_tensor=True)
    computed_value = torch.tensor([1, 2, 3])
    rm._computed = computed_value
    cache = ResultCollection._get_cache(rm, on_step=False)
    # `enable_graph=True` so no detach, identity works
    assert cache is computed_value


@pytest.mark.parametrize("floating_dtype", (torch.float, torch.double))
def test_metric_result_respects_dtype(floating_dtype):
    torch.set_default_dtype(floating_dtype)
    fixed_dtype = torch.long  # default by PyTorch

    metadata = _Metadata("foo", "bar")
    metadata.sync = _Sync()
    rm = ResultMetric(metadata, is_tensor=True)

    assert rm.value.dtype == floating_dtype
    assert rm.cumulated_batch_size.dtype == fixed_dtype

    # two fixed point numbers - should be converted
    value, batch_size = torch.tensor(2), torch.tensor(3)
    assert value.dtype == fixed_dtype
    with pytest.warns(
        UserWarning, match=rf"`self.log\('bar', ...\)` in your `foo` .* Converting it to {floating_dtype}"
    ):
        rm.update(value, batch_size)
    # floating and fixed
    rm.update(torch.tensor(4.0), torch.tensor(5))

    total = rm.compute()

    assert total == (2 * 3 + 4 * 5) / (5 + 3)
    assert total.dtype == floating_dtype

    # restore to avoid impacting other tests
    torch.set_default_dtype(torch.float)

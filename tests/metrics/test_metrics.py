import os
from typing import Any
import numpy as np
import pytest
import torch

import tests.base.develop_utils as tutils
from tests.base import EvalModelTemplate
from pytorch_lightning.metrics.metric import Metric, TensorMetric, NumpyMetric
from pytorch_lightning import Trainer


class DummyTensorMetric(TensorMetric):
    def __init__(self):
        super().__init__("dummy")

    def forward(self, input1, input2):
        assert isinstance(input1, torch.Tensor)
        assert isinstance(input2, torch.Tensor)
        return torch.tensor([1.0])


class DummyNumpyMetric(NumpyMetric):
    def __init__(self):
        super().__init__("dummy")

    def forward(self, input1, input2):
        assert isinstance(input1, np.ndarray)
        assert isinstance(input2, np.ndarray)
        return 1.0


class DummyTensorCollectionMetric(TensorMetric):
    def __init__(self):
        super().__init__("dummy")

    def forward(self, input1, input2):
        assert isinstance(input1, torch.Tensor)
        assert isinstance(input2, torch.Tensor)
        return 1.0, 2.0, 3.0, 4.0


@pytest.mark.parametrize("metric", [DummyTensorCollectionMetric()])
def test_collection_metric(metric: Metric):
    """ Test that metric.device, metric.dtype works for metric collection """
    input1, input2 = torch.tensor([1.0]), torch.tensor([2.0])

    def change_and_check_device_dtype(device, dtype):
        metric.to(device=device, dtype=dtype)

        metric_val = metric(input1, input2)
        assert not isinstance(metric_val, torch.Tensor)

        if device is not None:
            assert metric.device in [device, torch.device(device)]

        if dtype is not None:
            assert metric.dtype == dtype

    devices = [None, "cpu"]
    if torch.cuda.is_available():
        devices += ["cuda:0"]

    for device in devices:
        for dtype in [None, torch.float32, torch.float64]:
            change_and_check_device_dtype(device=device, dtype=dtype)

    if torch.cuda.is_available():
        metric.cuda(0)
        assert metric.device == torch.device("cuda", index=0)

    metric.cpu()
    assert metric.device == torch.device("cpu")

    metric.type(torch.int8)
    assert metric.dtype == torch.int8

    metric.float()
    assert metric.dtype == torch.float32

    metric.double()
    assert metric.dtype == torch.float64
    assert all(out.dtype == torch.float64 for out in metric(input1, input2))

    if torch.cuda.is_available():
        metric.cuda()
        metric.half()
        assert metric.dtype == torch.float16


@pytest.mark.parametrize(
    "metric",
    [
        DummyTensorMetric(),
        DummyNumpyMetric(),
    ],
)
def test_metric(metric: Metric):
    """ Test that metric.device, metric.dtype works for single metric"""
    input1, input2 = torch.tensor([1.0]), torch.tensor([2.0])

    def change_and_check_device_dtype(device, dtype):
        metric.to(device=device, dtype=dtype)

        metric_val = metric(input1, input2)
        assert isinstance(metric_val, torch.Tensor)

        if device is not None:
            assert metric.device in [device, torch.device(device)]
            assert metric_val.device in [device, torch.device(device)]

        if dtype is not None:
            assert metric.dtype == dtype
            assert metric_val.dtype == dtype

    devices = [None, "cpu"]
    if torch.cuda.is_available():
        devices += ["cuda:0"]

    for device in devices:
        for dtype in [None, torch.float32, torch.float64]:
            change_and_check_device_dtype(device=device, dtype=dtype)

    if torch.cuda.is_available():
        metric.cuda(0)
        assert metric.device == torch.device("cuda", index=0)
        assert metric(input1, input2).device == torch.device("cuda", index=0)

    metric.cpu()
    assert metric.device == torch.device("cpu")
    assert metric(input1, input2).device == torch.device("cpu")

    metric.float()
    assert metric.dtype == torch.float32
    assert metric(input1, input2).dtype == torch.float32

    metric.double()
    assert metric.dtype == torch.float64
    assert metric(input1, input2).dtype == torch.float64

    if torch.cuda.is_available():
        metric.cuda()
        metric.half()
        assert metric.dtype == torch.float16
        assert metric(input1, input2).dtype == torch.float16


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize("metric", [DummyTensorMetric, DummyNumpyMetric])
def test_model_pickable(tmpdir, metric: Metric):
    """Make sure that metrics are pickable by including into a model and running in multi-gpu mode"""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        gpus=[0, 1],
        distributed_backend="ddp_spawn",
    )

    model = EvalModelTemplate()
    model.metric = metric()
    model.training_step = model.training_step__using_metrics

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, "ddp model failed to complete"


@pytest.mark.parametrize("metric", [DummyTensorMetric(), DummyNumpyMetric()])
def test_saving_pickable(tmpdir, metric: Metric):
    """ Make sure that metrics are pickable by saving and loading them using torch """
    x, y = torch.randn(10,), torch.randn(
        10,
    )
    results_before_save = metric(x, y)

    # save metric
    save_path = os.path.join(tmpdir, "save_test.ckpt")
    torch.save(metric, save_path)

    # load metric
    new_metric = torch.load(save_path)
    results_after_load = new_metric(x, y)

    # Check metric value is the same
    assert results_before_save == results_after_load


def test_correct_call_order():
    """ Check that hooks are called in the expected order """

    class DummyMetric(Metric):
        def __init__(self):
            super().__init__("dummy")
            self.call_history = ["init"]

        @staticmethod
        def input_convert(self, data: Any):
            self.call_history.append("input_convert")
            return super(DummyMetric, self).input_convert(self, data)

        def forward(self, tensor1, tensor2):
            self.call_history.append("forward")
            return tensor1 - tensor2

        @staticmethod
        def output_convert(self, data: Any, output: Any):
            self.call_history.append("output_convert")
            return super(DummyMetric, self).output_convert(self, data, output)

        def ddp_sync(self, tensor: Any):
            self.call_history.append("ddp_sync")
            return super().ddp_sync(tensor)

        @staticmethod
        def ddp_reduce(self, data: Any, output: Any):
            self.call_history.append("ddp_reduce")
            return super(DummyMetric, self).ddp_reduce(self, data, output)

        def aggregate(self, *tensors: torch.Tensor) -> torch.Tensor:
            self.call_history.append("aggregate")
            return super().aggregate(*tensors)

        def reset(self):
            self.call_history.append("reset")
            return super().reset()

        @property
        def aggregated(self) -> torch.Tensor:
            self.call_history.append("aggregated")
            return super().aggregated

        @staticmethod
        def compute(self, data: Any, output: Any):
            self.call_history.append("compute")
            return super(DummyMetric, self).compute(self, data, output)

    metric = DummyMetric()
    assert metric.call_history == ["init"]
    result = metric(torch.tensor([2.0]), torch.tensor([1.0]))
    assert torch.allclose(result, torch.tensor(1.0))
    assert metric.call_history == [
        "init",
        "input_convert",
        "forward",
        "output_convert",
        "ddp_reduce",
        "ddp_sync",
        "aggregate",
        "compute"
    ]
    aggr = metric.aggregated
    assert metric.call_history == [
        "init",
        "input_convert",
        "forward",
        "output_convert",
        "ddp_reduce",
        "ddp_sync",
        "aggregate",
        "compute",
        "aggregated",
        "aggregate",
        "reset",
        "compute"
    ]
    assert torch.allclose(aggr, result)
    _ = metric(torch.tensor(2.0), torch.tensor(1.0))
    assert metric.call_history == [
        "init",
        "input_convert",
        "forward",
        "output_convert",
        "ddp_reduce",
        "ddp_sync",
        "aggregate",
        "compute",
        "aggregated",
        "aggregate",
        "reset",
        "compute",
        "input_convert",
        "forward",
        "output_convert",
        "ddp_reduce",
        "ddp_sync",
        "aggregate",
        "compute"
    ]

    metric = DummyMetric()
    _ = metric(torch.tensor([2.0]), torch.tensor([1.0]))
    _ = metric(torch.tensor([3.0]), torch.tensor([0.0]))

    aggregated = metric.aggregated

    assert torch.allclose(aggregated, torch.tensor(4.0))

    assert metric.call_history == [
        "init",
        "input_convert",
        "forward",
        "output_convert",
        "ddp_reduce",
        "ddp_sync",
        "aggregate",
        "compute",
        "input_convert",
        "forward",
        "output_convert",
        "ddp_reduce",
        "ddp_sync",
        "aggregate",
        "compute",
        "aggregated",
        "aggregate",
        "reset",
        "compute",
    ]

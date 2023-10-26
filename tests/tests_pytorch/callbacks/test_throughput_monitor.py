from unittest import mock
from unittest.mock import ANY, Mock, call

import pytest
import torch
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.throughput_monitor import ThroughputMonitor
from lightning.pytorch.demos.boring_classes import BoringModel

from tests_pytorch.helpers.runif import RunIf


@RunIf(min_torch="2.1")
def test_measure_flops():
    with torch.device("meta"):
        model = BoringModel()
        x = torch.randn(2, 32)
    model_fwd = lambda: model(x)

    fwd_flops = measure_flops(model, model_fwd)
    assert isinstance(fwd_flops, int)

    fwd_and_bwd_flops = measure_flops(model, model_fwd, model.loss)
    assert isinstance(fwd_and_bwd_flops, int)
    assert fwd_flops < fwd_and_bwd_flops


def test_throughput_monitor_fit(tmp_path):
    logger_mock = Mock()
    logger_mock.save_dir = tmp_path
    monitor = ThroughputMonitor(length_fn=lambda x: 2, batch_size_fn=lambda x: 3, window_size=4, separator="|")
    model = BoringModel()
    model.flops_per_batch = 10
    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=monitor,
        max_steps=5,
        log_every_n_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    # these timing results are meant to precisely match the `test_throughput_monitor` test in fabric
    timings = [0.0] + [0.5 + i for i in range(1, 6)]
    with mock.patch("lightning.pytorch.callbacks.throughput_monitor.get_available_flops", return_value=100), mock.patch(
        "time.perf_counter", side_effect=timings
    ):
        trainer.fit(model)

    # since limit_val_batches==0, we didn't init the validation throughput
    assert "validate" not in monitor._throughputs

    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"train|time": 1.5, "train|samples": 3, "epoch": 0}, step=0),
        call(metrics={"train|time": 2.5, "train|samples": 6, "epoch": 0}, step=1),
        call(metrics={"train|time": 3.5, "train|samples": 9, "epoch": 0}, step=2),
        call(
            metrics={
                "train|time": 4.5,
                "train|samples": 12,
                "train|device|batches_per_sec": 1.0,
                "train|device|samples_per_sec": 3.0,
                "train|device|items_per_sec": 6.0,
                "train|device|flops_per_sec": 10.0,
                "train|device|mfu": 0.1,
                "epoch": 0,
            },
            step=3,
        ),
        call(
            metrics={
                "train|time": 5.5,
                "train|samples": 15,
                "train|device|batches_per_sec": 1.0,
                "train|device|samples_per_sec": 3.0,
                "train|device|items_per_sec": 6.0,
                "train|device|flops_per_sec": 10.0,
                "train|device|mfu": 0.1,
                "epoch": 0,
            },
            step=4,
        ),
    ]


def test_throughput_monitor_fit_with_validation(tmp_path):
    logger_mock = Mock()
    logger_mock.save_dir = tmp_path
    monitor = ThroughputMonitor(batch_size_fn=lambda x: 1, window_size=2)
    model = BoringModel()
    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=monitor,
        max_steps=2,
        # validation runs in between the 2 training steps
        val_check_interval=1,
        log_every_n_steps=1,
        limit_val_batches=1,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    timings = [
        0,  # train t0
        7,  # train t1
        11,  # val t0
        13,  # val t1
        19,  # train t1
    ]
    with mock.patch("time.perf_counter", side_effect=timings):
        trainer.fit(model)

    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"train/time": 7, "train/samples": 1, "epoch": 0}, step=0),
        call(metrics={"validate/time": 13 - 11, "validate/samples": 1}, step=1),
        call(
            metrics={
                "train/time": 7 + (19 - 13),
                "train/samples": 2,
                "train/device/batches_per_sec": ANY,
                "train/device/samples_per_sec": ANY,
                "epoch": 0,
            },
            step=1,
        ),
    ]


def test_throughput_monitor_fit_no_length_fn(tmp_path):
    logger_mock = Mock()
    logger_mock.save_dir = tmp_path
    monitor = ThroughputMonitor(batch_size_fn=lambda x: 3, window_size=2)
    model = BoringModel()
    model.flops_per_batch = 33
    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=monitor,
        max_steps=3,
        log_every_n_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    with mock.patch("lightning.pytorch.callbacks.throughput_monitor.get_available_flops", return_value=100):
        trainer.fit(model)

    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"train/time": ANY, "train/samples": 3, "epoch": 0}, step=0),
        call(
            metrics={
                # using ANY to avoid issues with comparing floating numbers
                "train/time": ANY,
                "train/samples": 6,
                "train/device/batches_per_sec": ANY,
                "train/device/samples_per_sec": ANY,
                "train/device/flops_per_sec": ANY,
                "train/device/mfu": ANY,
                "epoch": 0,
            },
            step=1,
        ),
        call(
            metrics={
                "train/time": ANY,
                "train/samples": 9,
                "train/device/batches_per_sec": ANY,
                "train/device/samples_per_sec": ANY,
                "train/device/flops_per_sec": ANY,
                "train/device/mfu": ANY,
                "epoch": 0,
            },
            step=2,
        ),
    ]


def test_throughput_monitor_fit_gradient_accumulation(tmp_path):
    logger_mock = Mock()
    logger_mock.save_dir = tmp_path
    monitor = ThroughputMonitor(length_fn=lambda x: 2, batch_size_fn=lambda x: 3, window_size=4, separator="|")
    model = BoringModel()
    model.flops_per_batch = 10
    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=monitor,
        limit_train_batches=5,
        limit_val_batches=0,
        max_epochs=2,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    timings = [0.0] + [0.5 + i for i in range(1, 11)]
    with mock.patch("lightning.pytorch.callbacks.throughput_monitor.get_available_flops", return_value=100), mock.patch(
        "time.perf_counter", side_effect=timings
    ):
        trainer.fit(model)

    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"train|time": 2.5, "train|samples": 6, "epoch": 0}, step=0),
        call(
            metrics={
                "train|time": 4.5,
                "train|samples": 12,
                "train|device|batches_per_sec": 1.0,
                "train|device|samples_per_sec": 3.0,
                "train|device|items_per_sec": 6.0,
                "train|device|flops_per_sec": 10.0,
                "train|device|mfu": 0.1,
                "epoch": 0,
            },
            step=1,
        ),
        call(
            metrics={
                "train|time": 5.5,
                "train|samples": 15,
                "train|device|batches_per_sec": 1.0,
                "train|device|samples_per_sec": 3.0,
                "train|device|items_per_sec": 6.0,
                "train|device|flops_per_sec": 10.0,
                "train|device|mfu": 0.1,
                "epoch": 0,
            },
            step=2,
        ),
        call(
            metrics={
                "train|time": 7.5,
                "train|samples": 21,
                "train|device|batches_per_sec": 1.0,
                "train|device|samples_per_sec": 3.0,
                "train|device|items_per_sec": 6.0,
                "train|device|flops_per_sec": 10.0,
                "train|device|mfu": 0.1,
                "epoch": 1,
            },
            step=3,
        ),
        call(
            metrics={
                "train|time": 9.5,
                "train|samples": 27,
                "train|device|batches_per_sec": 1.0,
                "train|device|samples_per_sec": 3.0,
                "train|device|items_per_sec": 6.0,
                "train|device|flops_per_sec": 10.0,
                "train|device|mfu": 0.1,
                "epoch": 1,
            },
            step=4,
        ),
        call(
            metrics={
                "train|time": 10.5,
                "train|samples": 30,
                "train|device|batches_per_sec": 1.0,
                "train|device|samples_per_sec": 3.0,
                "train|device|items_per_sec": 6.0,
                "train|device|flops_per_sec": 10.0,
                "train|device|mfu": 0.1,
                "epoch": 1,
            },
            step=5,
        ),
    ]


@pytest.mark.parametrize("fn", ["validate", "test", "predict"])
def test_throughput_monitor_eval(tmp_path, fn):
    logger_mock = Mock()
    logger_mock.save_dir = tmp_path
    monitor = ThroughputMonitor(batch_size_fn=lambda x: 3, window_size=3, separator="|")
    model = BoringModel()
    model.flops_per_batch = 10
    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=monitor,
        limit_val_batches=6,
        limit_test_batches=6,
        limit_predict_batches=6,
        log_every_n_steps=3,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer_fn = getattr(trainer, fn)
    trainer_fn(model)

    assert logger_mock.log_metrics.mock_calls == [
        call(
            metrics={
                f"{fn}|time": ANY,
                f"{fn}|samples": 9,
                f"{fn}|device|batches_per_sec": ANY,
                f"{fn}|device|samples_per_sec": ANY,
                f"{fn}|device|flops_per_sec": ANY,
                f"{fn}|device|mfu": ANY,
            },
            step=3,
        ),
        call(
            metrics={
                f"{fn}|time": ANY,
                f"{fn}|samples": 18,
                f"{fn}|device|batches_per_sec": ANY,
                f"{fn}|device|samples_per_sec": ANY,
                f"{fn}|device|flops_per_sec": ANY,
                f"{fn}|device|mfu": ANY,
            },
            step=6,
        ),
    ]

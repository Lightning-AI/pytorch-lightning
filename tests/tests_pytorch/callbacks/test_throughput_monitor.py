from unittest import mock
from unittest.mock import ANY, Mock, call

import pytest
import torch

from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.throughput_monitor import ThroughputMonitor
from lightning.pytorch.demos.boring_classes import BoringModel


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
    monitor = ThroughputMonitor(length_fn=lambda x: 3 * 2, batch_size_fn=lambda x: 3, window_size=4, separator="|")
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
    with (
        mock.patch("lightning.pytorch.callbacks.throughput_monitor.get_available_flops", return_value=100),
        mock.patch("time.perf_counter", side_effect=timings),
    ):
        trainer.fit(model)

    # since limit_val_batches==0, we didn't init the validation throughput
    assert "validate" not in monitor._throughputs

    expected = {
        "train|device|batches_per_sec": 1.0,
        "train|device|samples_per_sec": 3.0,
        "train|device|items_per_sec": 6.0,
        "train|device|flops_per_sec": 10.0,
        "train|device|mfu": 0.1,
        "epoch": 0,
    }
    assert logger_mock.log_metrics.mock_calls == [
        call(
            metrics={"train|time": 1.5, "train|batches": 1, "train|samples": 3, "train|lengths": 6, "epoch": 0}, step=0
        ),
        call(
            metrics={"train|time": 2.5, "train|batches": 2, "train|samples": 6, "train|lengths": 12, "epoch": 0}, step=1
        ),
        call(
            metrics={"train|time": 3.5, "train|batches": 3, "train|samples": 9, "train|lengths": 18, "epoch": 0}, step=2
        ),
        call(
            metrics={**expected, "train|time": 4.5, "train|batches": 4, "train|samples": 12, "train|lengths": 24},
            step=3,
        ),
        call(
            metrics={**expected, "train|time": 5.5, "train|batches": 5, "train|samples": 15, "train|lengths": 30},
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
    timings = [0, 7, 11, 13, 19]  # train t0  # train t1  # val t0  # val t1  # train t1
    with mock.patch("time.perf_counter", side_effect=timings):
        trainer.fit(model)

    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"train/time": 7, "train/batches": 1, "train/samples": 1, "epoch": 0}, step=0),
        call(metrics={"validate/time": 13 - 11, "validate/batches": 1, "validate/samples": 1}, step=1),
        call(
            metrics={
                "train/time": 7 + (19 - 13),
                "train/batches": 2,
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

    expected = {
        # using ANY to avoid issues with comparing floating numbers
        "train/time": ANY,
        "train/device/batches_per_sec": ANY,
        "train/device/samples_per_sec": ANY,
        "train/device/flops_per_sec": ANY,
        "train/device/mfu": ANY,
        "epoch": 0,
    }
    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"train/time": ANY, "train/batches": 1, "train/samples": 3, "epoch": 0}, step=0),
        call(metrics={**expected, "train/batches": 2, "train/samples": 6}, step=1),
        call(metrics={**expected, "train/batches": 3, "train/samples": 9}, step=2),
    ]


@pytest.mark.parametrize("log_every_n_steps", [1, 3])
def test_throughput_monitor_fit_gradient_accumulation(log_every_n_steps, tmp_path):
    logger_mock = Mock()
    logger_mock.save_dir = tmp_path
    monitor = ThroughputMonitor(length_fn=lambda x: 3 * 2, batch_size_fn=lambda x: 3, window_size=4, separator="|")
    model = BoringModel()
    model.flops_per_batch = 10

    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=monitor,
        limit_train_batches=5,
        limit_val_batches=0,
        max_epochs=2,
        log_every_n_steps=log_every_n_steps,
        accumulate_grad_batches=2,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    timings = [0.0] + [0.5 + i for i in range(1, 11)]
    with (
        mock.patch("lightning.pytorch.callbacks.throughput_monitor.get_available_flops", return_value=100),
        mock.patch("time.perf_counter", side_effect=timings),
    ):
        trainer.fit(model)

    expected = {
        "train|device|batches_per_sec": 1.0,
        "train|device|samples_per_sec": 3.0,
        "train|device|items_per_sec": 6.0,
        "train|device|flops_per_sec": 10.0,
        "train|device|mfu": 0.1,
    }

    all_log_calls = [
        call(
            metrics={
                # The very first batch doesn't have the *_per_sec metrics yet
                **(expected if log_every_n_steps > 1 else {}),
                "train|time": 2.5,
                "train|batches": 2,
                "train|samples": 6,
                "train|lengths": 12,
                "epoch": 0,
            },
            step=0,
        ),
        call(
            metrics={
                **expected,
                "train|time": 4.5,
                "train|batches": 4,
                "train|samples": 12,
                "train|lengths": 24,
                "epoch": 0,
            },
            step=1,
        ),
        call(
            metrics={
                **expected,
                "train|time": 5.5,
                "train|batches": 5,
                "train|samples": 15,
                "train|lengths": 30,
                "epoch": 0,
            },
            step=2,
        ),
        call(
            metrics={
                **expected,
                "train|time": 7.5,
                "train|batches": 7,
                "train|samples": 21,
                "train|lengths": 42,
                "epoch": 1,
            },
            step=3,
        ),
        call(
            metrics={
                **expected,
                "train|time": 9.5,
                "train|batches": 9,
                "train|samples": 27,
                "train|lengths": 54,
                "epoch": 1,
            },
            step=4,
        ),
        call(
            metrics={
                **expected,
                "train|time": 10.5,
                "train|batches": 10,
                "train|samples": 30,
                "train|lengths": 60,
                "epoch": 1,
            },
            step=5,
        ),
    ]
    expected_log_calls = all_log_calls[(log_every_n_steps - 1) :: log_every_n_steps]
    assert logger_mock.log_metrics.mock_calls == expected_log_calls


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

    with mock.patch("lightning.pytorch.callbacks.throughput_monitor.get_available_flops", return_value=100):
        trainer_fn(model)
        trainer_fn(model)

    expected = {
        f"{fn}|time": ANY,
        f"{fn}|device|batches_per_sec": ANY,
        f"{fn}|device|samples_per_sec": ANY,
        f"{fn}|device|flops_per_sec": ANY,
        f"{fn}|device|mfu": ANY,
    }
    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={**expected, f"{fn}|batches": 3, f"{fn}|samples": 9}, step=3),
        call(metrics={**expected, f"{fn}|batches": 6, f"{fn}|samples": 18}, step=6),
        # the step doesn't repeat
        call(metrics={**expected, f"{fn}|batches": 9, f"{fn}|samples": 27}, step=9),
        call(metrics={**expected, f"{fn}|batches": 12, f"{fn}|samples": 36}, step=12),
    ]


def test_throughput_monitor_variable_batch_size(tmp_path):
    """Test that ThroughputMonitor correctly handles variable batch sizes."""
    logger_mock = Mock()
    logger_mock.save_dir = tmp_path

    # Simulate variable batch sizes by tracking calls
    batch_sizes = [1, 3, 2, 1, 4]
    call_count = [0]

    def variable_batch_size_fn(batch):
        # Return the predefined batch size for this call
        current_batch_size = batch_sizes[call_count[0] % len(batch_sizes)]
        call_count[0] += 1
        return current_batch_size

    monitor = ThroughputMonitor(batch_size_fn=variable_batch_size_fn, window_size=5, separator="|")

    model = BoringModel()
    model.flops_per_batch = 10

    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=monitor,
        max_steps=len(batch_sizes),
        log_every_n_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    timings = [0.0] + [i * 0.1 for i in range(1, len(batch_sizes) + 1)]

    with (
        mock.patch("lightning.pytorch.callbacks.throughput_monitor.get_available_flops", return_value=100),
        mock.patch("time.perf_counter", side_effect=timings),
    ):
        trainer.fit(model)

    log_calls = logger_mock.log_metrics.call_args_list
    assert len(log_calls) == len(batch_sizes)

    # Expected cumulative samples: 1, 4 (1+3), 6 (4+2), 7 (6+1), 11 (7+4)
    expected_cumulative_samples = [1, 4, 6, 7, 11]

    for i, log_call in enumerate(log_calls):
        metrics = log_call.kwargs["metrics"] if "metrics" in log_call.kwargs else log_call.args[0]
        expected_samples = expected_cumulative_samples[i]
        assert metrics["train|samples"] == expected_samples, (
            f"Step {i}: expected {expected_samples}, got {metrics['train|samples']}"
        )
        assert metrics["train|batches"] == i + 1, f"Step {i}: expected batches {i + 1}, got {metrics['train|batches']}"


def test_throughput_monitor_variable_batch_size_with_validation(tmp_path):
    """Test variable batch sizes with validation to ensure stage isolation."""
    logger_mock = Mock()
    logger_mock.save_dir = tmp_path

    train_batch_sizes = [2, 1, 3]
    val_batch_sizes = [1, 2]
    train_call_count = [0]
    val_call_count = [0]

    def variable_batch_size_fn(batch):
        if hasattr(batch, "size") and batch.size(0) > 0:
            if train_call_count[0] < len(train_batch_sizes):
                current_batch_size = train_batch_sizes[train_call_count[0]]
                train_call_count[0] += 1
                return current_batch_size
            current_batch_size = val_batch_sizes[val_call_count[0] % len(val_batch_sizes)]
            val_call_count[0] += 1
            return current_batch_size
        return 1

    monitor = ThroughputMonitor(batch_size_fn=variable_batch_size_fn, window_size=3)
    model = BoringModel()

    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=monitor,
        max_steps=len(train_batch_sizes),
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=2,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    with mock.patch("lightning.pytorch.callbacks.throughput_monitor.get_available_flops", return_value=100):
        trainer.fit(model)

    # Verify that both training and validation metrics were logged
    log_calls = logger_mock.log_metrics.call_args_list
    train_calls = [call for call in log_calls if "train/" in str(call) or "train|" in str(call)]
    val_calls = [call for call in log_calls if "validate/" in str(call) or "validate|" in str(call)]

    assert len(train_calls) > 0, "Expected training metrics to be logged"
    assert len(val_calls) > 0, "Expected validation metrics to be logged"
    train_samples = []
    for train_call in train_calls:
        metrics = train_call.kwargs.get("metrics", train_call.args[0] if train_call.args else {})
        if "train/samples" in metrics:
            train_samples.append(metrics["train/samples"])
        elif "train|samples" in metrics:
            train_samples.append(metrics["train|samples"])


def test_throughput_monitor_validation_with_many_epochs(tmp_path):
    """Ensure ThroughputMonitor handles many epochs with validation and time increases monotonically."""

    logger_mock = Mock()
    logger_mock.save_dir = tmp_path
    monitor = ThroughputMonitor(batch_size_fn=lambda x: 1)
    model = BoringModel()
    model.flops_per_batch = 10
    num_epochs = 100

    trainer = Trainer(
        devices=1,
        logger=logger_mock,
        callbacks=[monitor],
        max_epochs=num_epochs,
        limit_train_batches=2,
        limit_val_batches=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    timings = []
    t = 0.0
    for _ in range(num_epochs):
        timings += [
            t,  # train batch 1 start
            t + 3.0,  # train batch 1 end and start batch 2
            t + 6.0,  # train batch 2 end
            t + 7.0,  # val start
            t + 8.0,  # val end
        ]
        t += 10.0

    with mock.patch("time.perf_counter", side_effect=timings):
        try:
            trainer.fit(model)
        except Exception as e:
            pytest.fail(f"ThroughputMonitor raised an unexpected exception: {e}")

    start_train_timings_idx, end_train_timings_idx = 0, 1
    batch_num = 1
    cur_train = timings[end_train_timings_idx] - timings[start_train_timings_idx]
    for c in logger_mock.log_metrics.mock_calls:
        metrics = getattr(c, "kwargs", None) or {}
        metrics = metrics.get("metrics", metrics)
        for k, v in metrics.items():
            if k.endswith("train/time"):
                assert v == cur_train, f"Expected train/time {cur_train}, got {v}"
                if batch_num == 1:
                    start_train_timings_idx += 1
                    end_train_timings_idx += 1
                    batch_num = 2
                else:
                    start_train_timings_idx += 3
                    end_train_timings_idx += 3
                    batch_num = 1
                if end_train_timings_idx < len(timings):
                    cur_train += timings[end_train_timings_idx] - timings[start_train_timings_idx]

# Copyright The Lightning AI team.
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
from functools import partial
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher
from lightning.pytorch.trainer.connectors.logger_connector.result import (
    _assert_sync_dist_metric_keys_consistency,
    _ResultCollection,
    _Sync,
)
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.models.test_tpu import wrap_launch_function


def spawn_launch(fn, parallel_devices):
    # TODO: the accelerator and cluster_environment should be optional to just launch processes, but this requires lazy
    # initialization to be implemented
    device_to_accelerator = {"cuda": CUDAAccelerator, "mps": MPSAccelerator, "cpu": CPUAccelerator}
    accelerator_cls = device_to_accelerator[parallel_devices[0].type]
    strategy = DDPStrategy(
        accelerator=accelerator_cls(),
        parallel_devices=parallel_devices,
        cluster_environment=LightningEnvironment(),
        start_method="spawn",
    )
    launcher = _MultiProcessingLauncher(strategy=strategy)
    wrapped = partial(wrap_launch_function, fn, strategy)
    return launcher.launch(wrapped, strategy)


def result_reduce_ddp_fn(strategy):
    tensor = torch.tensor([1.0])
    sync = _Sync(strategy.reduce, _should=True, _op="SUM")
    actual = sync(tensor)
    assert actual.item() == dist.get_world_size()


# flaky with "torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGABRT"
@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_result_reduce_ddp():
    spawn_launch(result_reduce_ddp_fn, [torch.device("cpu")] * 2)


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=False)
def test_assert_sync_dist_metric_keys_consistency_not_initialized(_):
    """Test that _assert_sync_dist_metric_keys_consistency returns early when dist is not initialized."""
    # Should not raise even with mismatched keys since dist is not initialized
    _assert_sync_dist_metric_keys_consistency(["key_a"], "training_step", None)


@patch("torch.distributed.is_available", return_value=False)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_assert_sync_dist_metric_keys_consistency_dist_not_available(_, __):
    """Test that _assert_sync_dist_metric_keys_consistency returns early when dist is not available."""
    # Should not raise even with mismatched keys since dist is not available
    _assert_sync_dist_metric_keys_consistency(["key_a"], "training_step", None)


@patch("torch.distributed.get_world_size", return_value=1)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_assert_sync_dist_metric_keys_consistency_single_process(_, __, ___):
    """Test that _assert_sync_dist_metric_keys_consistency returns early with world_size=1."""
    # Should not raise with single process
    _assert_sync_dist_metric_keys_consistency(["key_a"], "training_step", None)


def _sync_dist_keys_consistency_match_fn(strategy):
    """Function to test that matching keys across ranks doesn't raise an error."""
    # All ranks have the same keys
    keys = ["training_step.loss", "training_step.acc"]
    # This should not raise
    _assert_sync_dist_metric_keys_consistency(keys, "training_step", None)


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_assert_sync_dist_metric_keys_consistency_match():
    """Test that _assert_sync_dist_metric_keys_consistency doesn't raise when keys match."""
    spawn_launch(_sync_dist_keys_consistency_match_fn, [torch.device("cpu")] * 2)


def _sync_dist_keys_consistency_mismatch_fn(strategy):
    """Function to test that mismatched keys across ranks raises an error."""
    rank = dist.get_rank()
    keys = ["training_step.metric_a"] if rank == 0 else ["training_step.metric_b"]

    with pytest.raises(MisconfigurationException) as excinfo:
        _assert_sync_dist_metric_keys_consistency(keys, "training_step", None)

    message = str(excinfo.value)
    assert "sync_dist=True" in message
    assert "Detected a mismatch" in message
    assert "training_step.metric_a" in message
    assert "training_step.metric_b" in message


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_assert_sync_dist_metric_keys_consistency_mismatch():
    """Test that _assert_sync_dist_metric_keys_consistency raises when keys mismatch."""
    spawn_launch(_sync_dist_keys_consistency_mismatch_fn, [torch.device("cpu")] * 2)


def _sync_dist_keys_consistency_order_mismatch_fn(strategy):
    """Function to test that keys in different order across ranks raises an error."""
    rank = dist.get_rank()
    if rank == 0:
        keys = ["training_step.metric_a", "training_step.metric_b"]
    else:
        keys = ["training_step.metric_b", "training_step.metric_a"]

    with pytest.raises(MisconfigurationException) as excinfo:
        _assert_sync_dist_metric_keys_consistency(keys, "training_step", None)

    message = str(excinfo.value)
    assert "sync_dist=True" in message
    assert "Detected a mismatch" in message


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_assert_sync_dist_metric_keys_consistency_order_mismatch():
    """Test that _assert_sync_dist_metric_keys_consistency raises when key order differs."""
    spawn_launch(_sync_dist_keys_consistency_order_mismatch_fn, [torch.device("cpu")] * 2)


def _sync_dist_keys_consistency_empty_keys_fn(strategy):
    """Function to test that empty keys across all ranks doesn't raise an error."""
    keys: list[str] = []
    # Empty keys should not raise (no metrics to sync)
    _assert_sync_dist_metric_keys_consistency(keys, "training_step", None)


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_assert_sync_dist_metric_keys_consistency_empty_keys():
    """Test that _assert_sync_dist_metric_keys_consistency doesn't raise with empty keys on all ranks."""
    spawn_launch(_sync_dist_keys_consistency_empty_keys_fn, [torch.device("cpu")] * 2)


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=False)
def test_sync_on_step_metrics_not_distributed(_):
    """Test that sync_on_step_metrics returns early when not in distributed mode."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)
    # Should not raise, just return early
    result.sync_on_step_metrics()


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=False)
def test_sync_on_epoch_metrics_not_distributed(_):
    """Test that sync_on_epoch_metrics returns early when not in distributed mode."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=True)
    # Should not raise, just return early
    result.sync_on_epoch_metrics()


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._assert_sync_dist_metric_keys_consistency")
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_no_items_to_sync(_, mock_assert):
    """Test that sync_on_step_metrics returns early when no items need syncing."""
    result = _ResultCollection(training=True)
    # Log without sync_dist=True
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=False)
    result.sync_on_step_metrics()
    # Should not call the validation since there are no items to sync
    mock_assert.assert_not_called()


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._assert_sync_dist_metric_keys_consistency")
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_epoch_metrics_no_items_to_sync(_, mock_assert):
    """Test that sync_on_epoch_metrics returns early when no items need syncing."""
    result = _ResultCollection(training=True)
    # Log without sync_dist=True
    result.log("training_step", "loss", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=False)
    result.sync_on_epoch_metrics()
    # Should not call the validation since there are no items to sync
    mock_assert.assert_not_called()


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._assert_sync_dist_metric_keys_consistency")
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_rank_zero_only_skipped(_, mock_assert):
    """Test that rank_zero_only metrics are skipped from sync validation."""
    result = _ResultCollection(training=True)
    # Log with rank_zero_only=True - should be skipped from sync validation
    result.log(
        "training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True, rank_zero_only=True
    )
    result.sync_on_step_metrics()
    # Should not call the validation since rank_zero_only metrics are skipped
    mock_assert.assert_not_called()


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._assert_sync_dist_metric_keys_consistency")
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_epoch_metrics_rank_zero_only_skipped(_, mock_assert):
    """Test that rank_zero_only metrics are skipped from sync validation."""
    result = _ResultCollection(training=True)
    # Log with rank_zero_only=True - should be skipped from sync validation
    result.log(
        "training_step", "loss", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True
    )
    result.sync_on_epoch_metrics()
    # Should not call the validation since rank_zero_only metrics are skipped
    mock_assert.assert_not_called()


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=False)
def test_result_metric_deferred_sync_behavior(_):
    """Test that on_step metrics defer sync until sync_on_step_metrics is called."""
    result = _ResultCollection(training=True)

    # Log with on_step=True and sync_dist=True
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)

    # Before sync_on_step_metrics is called, forward cache should be set but not synced
    loss_metric = result["training_step.loss"]
    assert loss_metric._forward_cache is not None, "Forward cache should be set"
    assert loss_metric._forward_cache_synced is False, "Forward cache should not be synced yet"

    # In non-distributed mode, sync_on_step_metrics returns early
    result.sync_on_step_metrics()
    # Still not synced because we're not in distributed mode
    assert loss_metric._forward_cache_synced is False


def _sync_on_step_metrics_ddp_fn(strategy):
    """Function to test sync_on_step_metrics in DDP mode with consistent keys."""
    result = _ResultCollection(training=True)

    # All ranks log the same keys
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)
    result.log("training_step", "acc", torch.tensor(0.9), on_step=True, on_epoch=False, sync_dist=True)

    # Before sync, forward cache should be set but not synced
    loss_metric = result["training_step.loss"]
    assert loss_metric._forward_cache is not None, "Forward cache should be set"
    assert loss_metric._forward_cache_synced is False, "Forward cache should not be synced yet"

    # This should not raise since all ranks have the same keys
    result.sync_on_step_metrics()

    # Verify the forward cache was synced
    assert loss_metric._forward_cache_synced is True


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_sync_on_step_metrics_ddp():
    """Test sync_on_step_metrics works correctly in DDP with consistent keys."""
    spawn_launch(_sync_on_step_metrics_ddp_fn, [torch.device("cpu")] * 2)


def _sync_on_epoch_metrics_ddp_fn(strategy):
    """Function to test sync_on_epoch_metrics in DDP mode with consistent keys."""
    result = _ResultCollection(training=True)

    # All ranks log the same keys
    result.log("training_step", "loss", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=True)
    result.log("training_step", "acc", torch.tensor(0.9), on_step=False, on_epoch=True, sync_dist=True)

    # This should not raise since all ranks have the same keys
    result.sync_on_epoch_metrics()

    # Verify compute was called (which sets _computed)
    loss_metric = result["training_step.loss"]
    assert loss_metric._computed is not None


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_sync_on_epoch_metrics_ddp():
    """Test sync_on_epoch_metrics works correctly in DDP with consistent keys."""
    spawn_launch(_sync_on_epoch_metrics_ddp_fn, [torch.device("cpu")] * 2)


def _sync_on_step_metrics_mismatch_ddp_fn(strategy):
    """Function to test sync_on_step_metrics raises with mismatched keys."""
    rank = dist.get_rank()
    result = _ResultCollection(training=True)

    # Different ranks log different keys
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)
    if rank == 0:
        result.log("training_step", "metric_a", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)
    else:
        result.log("training_step", "metric_b", torch.tensor(2.0), on_step=True, on_epoch=False, sync_dist=True)

    with pytest.raises(MisconfigurationException) as excinfo:
        result.sync_on_step_metrics()

    message = str(excinfo.value)
    assert "sync_dist=True" in message
    assert "Detected a mismatch" in message


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_sync_on_step_metrics_mismatch_ddp():
    """Test sync_on_step_metrics raises with mismatched keys in DDP."""
    spawn_launch(_sync_on_step_metrics_mismatch_ddp_fn, [torch.device("cpu")] * 2)


def _sync_on_epoch_metrics_mismatch_ddp_fn(strategy):
    """Function to test sync_on_epoch_metrics raises with mismatched keys."""
    rank = dist.get_rank()
    result = _ResultCollection(training=True)

    # Different ranks log different keys
    result.log("training_step", "loss", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=True)
    if rank == 0:
        result.log("training_step", "metric_a", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=True)
    else:
        result.log("training_step", "metric_b", torch.tensor(2.0), on_step=False, on_epoch=True, sync_dist=True)

    with pytest.raises(MisconfigurationException) as excinfo:
        result.sync_on_epoch_metrics()

    message = str(excinfo.value)
    assert "sync_dist=True" in message
    assert "Detected a mismatch" in message


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_sync_on_epoch_metrics_mismatch_ddp():
    """Test sync_on_epoch_metrics raises with mismatched keys in DDP."""
    spawn_launch(_sync_on_epoch_metrics_mismatch_ddp_fn, [torch.device("cpu")] * 2)

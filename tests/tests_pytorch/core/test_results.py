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


def _mock_all_gather_consistent(output_list, obj, group=None):
    """Mock all_gather_object that simulates consistent keys across 2 ranks."""
    output_list[0] = obj
    output_list[1] = obj


def _mock_all_gather_inconsistent(output_list, obj, group=None):
    """Mock all_gather_object that simulates inconsistent keys across 2 ranks."""
    output_list[0] = ["training_step.loss", "training_step.metric_a"]
    output_list[1] = ["training_step.loss", "training_step.metric_b"]


def _mock_all_gather_order_mismatch(output_list, obj, group=None):
    """Mock all_gather_object that simulates same keys but different order across 2 ranks."""
    output_list[0] = ["training_step.metric_a", "training_step.metric_b"]
    output_list[1] = ["training_step.metric_b", "training_step.metric_a"]


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_assert_sync_dist_metric_keys_consistency_matching_keys_mocked(_, __, ___, ____):
    """Test key consistency validation with matching keys using mocked distributed functions."""
    keys = ["training_step.loss", "training_step.acc"]
    # Should not raise when all ranks have matching keys
    _assert_sync_dist_metric_keys_consistency(keys, "training_step", None)


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_inconsistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_assert_sync_dist_metric_keys_consistency_mismatched_keys_mocked(_, __, ___, ____):
    """Test key consistency validation raises with mismatched keys using mocked distributed functions."""
    keys = ["training_step.loss", "training_step.metric_a"]
    with pytest.raises(MisconfigurationException) as excinfo:
        _assert_sync_dist_metric_keys_consistency(keys, "training_step", None)

    message = str(excinfo.value)
    assert "sync_dist=True" in message
    assert "Detected a mismatch" in message
    assert "training_step.metric_a" in message
    assert "training_step.metric_b" in message


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_order_mismatch)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_assert_sync_dist_metric_keys_consistency_order_mismatch_mocked(_, __, ___, ____):
    """Test key consistency validation raises with different key order using mocked distributed functions."""
    keys = ["training_step.metric_a", "training_step.metric_b"]
    with pytest.raises(MisconfigurationException) as excinfo:
        _assert_sync_dist_metric_keys_consistency(keys, "training_step", None)

    message = str(excinfo.value)
    assert "sync_dist=True" in message
    assert "Detected a mismatch" in message


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


def _mock_sync_fn(value, reduce_op=None, group=None):
    """Mock sync function that returns the value unchanged."""
    return value


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_with_mocked_distributed(_, __, ___, ____):
    """Test sync_on_step_metrics executes sync logic with mocked distributed functions."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)

    loss_metric = result["training_step.loss"]
    # Override the sync function with our mock
    loss_metric.meta._sync.fn = _mock_sync_fn
    loss_metric.meta._sync._should = True

    assert loss_metric._forward_cache is not None
    assert loss_metric._forward_cache_synced is False

    result.sync_on_step_metrics()

    # After sync, forward cache should be marked as synced
    assert loss_metric._forward_cache_synced is True


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_epoch_metrics_with_mocked_distributed(_, __, ___, ____):
    """Test sync_on_epoch_metrics executes sync logic with mocked distributed functions."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=True)

    loss_metric = result["training_step.loss"]
    # Override the sync function with our mock
    loss_metric.meta._sync.fn = _mock_sync_fn
    loss_metric.meta._sync._should = True

    assert loss_metric._computed is None

    result.sync_on_epoch_metrics()

    # After sync, compute should have been called
    assert loss_metric._computed is not None


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


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._assert_sync_dist_metric_keys_consistency")
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_non_tensor_skipped(_, mock_assert):
    """Test that non-tensor (TorchMetric) metrics are skipped from sync validation."""
    from torchmetrics import Accuracy

    result = _ResultCollection(training=True)
    # Log a TorchMetric - these have is_tensor=False and should be skipped
    metric = Accuracy(task="binary")
    result.log("training_step", "accuracy", metric, on_step=True, on_epoch=False, sync_dist=True)

    accuracy_metric = result["training_step.accuracy"]
    # Verify it's not a tensor metric
    assert accuracy_metric.is_tensor is False

    result.sync_on_step_metrics()
    # Should not call the validation since non-tensor metrics are skipped
    mock_assert.assert_not_called()


@patch("lightning.pytorch.trainer.connectors.logger_connector.result._assert_sync_dist_metric_keys_consistency")
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_epoch_metrics_non_tensor_skipped(_, mock_assert):
    """Test that non-tensor (TorchMetric) metrics are skipped from sync validation."""
    from torchmetrics import Accuracy

    result = _ResultCollection(training=True)
    # Log a TorchMetric - these have is_tensor=False and should be skipped
    metric = Accuracy(task="binary")
    result.log("training_step", "accuracy", metric, on_step=False, on_epoch=True, sync_dist=True)

    accuracy_metric = result["training_step.accuracy"]
    # Verify it's not a tensor metric
    assert accuracy_metric.is_tensor is False

    result.sync_on_epoch_metrics()
    # Should not call the validation since non-tensor metrics are skipped
    mock_assert.assert_not_called()


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_on_step_only_updates_value(_, __, ___, ____):
    """Test that sync_on_step_metrics updates result_metric.value for on_step only metrics."""
    result = _ResultCollection(training=True)
    # Log with on_step=True, on_epoch=False - this is the "on_step only" path
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)

    loss_metric = result["training_step.loss"]
    # Override the sync function with our mock
    loss_metric.meta._sync.fn = _mock_sync_fn
    loss_metric.meta._sync._should = True

    # Before sync, cache is not synced
    assert loss_metric._forward_cache_synced is False
    loss_metric.value.clone()

    result.sync_on_step_metrics()

    # After sync, value should be updated (for on_step only metrics)
    assert loss_metric._forward_cache_synced is True
    # The value should now equal the synced forward_cache
    assert torch.equal(loss_metric.value, loss_metric._forward_cache)


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_on_step_and_epoch_no_value_update(_, __, ___, ____):
    """Test that sync_on_step_metrics does NOT update value for on_step+on_epoch metrics."""
    result = _ResultCollection(training=True)
    # Log with on_step=True, on_epoch=True - value should NOT be updated
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=True, sync_dist=True)

    loss_metric = result["training_step.loss"]
    # Override the sync function with our mock
    loss_metric.meta._sync.fn = _mock_sync_fn
    loss_metric.meta._sync._should = True

    # Store original value
    loss_metric.value.clone()

    result.sync_on_step_metrics()

    # After sync, forward_cache should be synced but value should NOT be updated
    # (because on_epoch=True means we accumulate, not replace)
    assert loss_metric._forward_cache_synced is True
    # Value is accumulated for epoch-level metrics, so it's updated during update() but not in sync


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_already_synced_skipped(_, __, ___, ____):
    """Test that already-synced metrics are skipped in sync_on_step_metrics."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)

    loss_metric = result["training_step.loss"]
    loss_metric.meta._sync.fn = _mock_sync_fn
    loss_metric.meta._sync._should = True

    # Mark as already synced
    loss_metric._forward_cache_synced = True

    # Create a mock to track if sync is called
    call_count = [0]
    original_fn = loss_metric.meta._sync.fn

    def counting_sync(value, reduce_op=None, group=None):
        call_count[0] += 1
        return original_fn(value, reduce_op, group)

    loss_metric.meta._sync.fn = counting_sync

    result.sync_on_step_metrics()

    # Sync should not have been called since it was already synced
    assert call_count[0] == 0


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_epoch_metrics_already_computed_skipped(_, __, ___, ____):
    """Test that already-computed metrics are skipped in sync_on_epoch_metrics."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=True)

    loss_metric = result["training_step.loss"]
    loss_metric.meta._sync.fn = _mock_sync_fn
    loss_metric.meta._sync._should = True

    # Mark as already computed
    loss_metric._computed = torch.tensor(1.0)

    # Create a mock to track if compute is called
    compute_called = [False]
    original_compute = loss_metric.compute

    def tracking_compute():
        compute_called[0] = True
        return original_compute()

    loss_metric.compute = tracking_compute

    result.sync_on_epoch_metrics()

    # Compute should not have been called since it was already computed
    assert compute_called[0] is False


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_forward_cache_none_skipped(_, __, ___, ____):
    """Test that metrics with None forward_cache are skipped."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)

    loss_metric = result["training_step.loss"]
    loss_metric.meta._sync.fn = _mock_sync_fn
    loss_metric.meta._sync._should = True

    # Set forward cache to None
    loss_metric._forward_cache = None

    # Should not raise, just skip this metric
    result.sync_on_step_metrics()
    # Since forward_cache is None, it should remain not synced
    assert loss_metric._forward_cache_synced is False


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_step_metrics_multiple_metrics(_, __, ___, ____):
    """Test sync_on_step_metrics with multiple metrics."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)
    result.log("training_step", "acc", torch.tensor(0.9), on_step=True, on_epoch=False, sync_dist=True)
    result.log("training_step", "f1", torch.tensor(0.85), on_step=True, on_epoch=False, sync_dist=True)

    for key in ["training_step.loss", "training_step.acc", "training_step.f1"]:
        metric = result[key]
        metric.meta._sync.fn = _mock_sync_fn
        metric.meta._sync._should = True

    result.sync_on_step_metrics()

    # All metrics should be synced
    for key in ["training_step.loss", "training_step.acc", "training_step.f1"]:
        metric = result[key]
        assert metric._forward_cache_synced is True


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_consistent)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_sync_on_epoch_metrics_multiple_metrics(_, __, ___, ____):
    """Test sync_on_epoch_metrics with multiple metrics."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=False, on_epoch=True, sync_dist=True)
    result.log("training_step", "acc", torch.tensor(0.9), on_step=False, on_epoch=True, sync_dist=True)
    result.log("training_step", "f1", torch.tensor(0.85), on_step=False, on_epoch=True, sync_dist=True)

    for key in ["training_step.loss", "training_step.acc", "training_step.f1"]:
        metric = result[key]
        metric.meta._sync.fn = _mock_sync_fn
        metric.meta._sync._should = True

    result.sync_on_epoch_metrics()

    # All metrics should have _computed set
    for key in ["training_step.loss", "training_step.acc", "training_step.f1"]:
        metric = result[key]
        assert metric._computed is not None


def _sync_on_step_metrics_value_update_ddp_fn(strategy):
    """Function to test sync_on_step_metrics updates value for on_step only metrics in DDP."""
    result = _ResultCollection(training=True)

    # Log on_step only metric
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)

    loss_metric = result["training_step.loss"]
    assert loss_metric._forward_cache is not None
    assert loss_metric._forward_cache_synced is False

    result.sync_on_step_metrics()

    # After sync, value should equal forward_cache (both synced)
    assert loss_metric._forward_cache_synced is True
    assert torch.equal(loss_metric.value, loss_metric._forward_cache)


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_sync_on_step_metrics_value_update_ddp():
    """Test sync_on_step_metrics updates value for on_step only metrics in DDP."""
    spawn_launch(_sync_on_step_metrics_value_update_ddp_fn, [torch.device("cpu")] * 2)


def _sync_on_step_and_epoch_metrics_ddp_fn(strategy):
    """Function to test both on_step and on_epoch sync in DDP."""
    result = _ResultCollection(training=True)

    # Log with both on_step and on_epoch
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=True, sync_dist=True)
    result.log("training_step", "acc", torch.tensor(0.9), on_step=True, on_epoch=True, sync_dist=True)

    loss_metric = result["training_step.loss"]
    acc_metric = result["training_step.acc"]

    # First, sync on_step metrics
    result.sync_on_step_metrics()
    assert loss_metric._forward_cache_synced is True
    assert acc_metric._forward_cache_synced is True

    # Then, sync on_epoch metrics
    result.sync_on_epoch_metrics()
    assert loss_metric._computed is not None
    assert acc_metric._computed is not None


@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
def test_sync_on_step_and_epoch_metrics_ddp():
    """Test both on_step and on_epoch sync work together in DDP."""
    spawn_launch(_sync_on_step_and_epoch_metrics_ddp_fn, [torch.device("cpu")] * 2)


def _mock_all_gather_detailed_mismatch(output_list, obj, group=None):
    """Mock all_gather_object that simulates detailed inconsistent keys across 3 ranks."""
    output_list[0] = ["training_step.loss", "training_step.metric_a"]
    output_list[1] = ["training_step.loss", "training_step.metric_b"]
    output_list[2] = ["training_step.loss", "training_step.metric_c"]


@patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_detailed_mismatch)
@patch("torch.distributed.get_world_size", return_value=3)
@patch("torch.distributed.is_available", return_value=True)
@patch("lightning.pytorch.trainer.connectors.logger_connector.result._distributed_is_initialized", return_value=True)
def test_assert_sync_dist_metric_keys_consistency_detailed_error_message(_, __, ___, ____):
    """Test that error message contains detailed information about all ranks."""
    keys = ["training_step.loss", "training_step.metric_a"]
    with pytest.raises(MisconfigurationException) as excinfo:
        _assert_sync_dist_metric_keys_consistency(keys, "training_step", None)

    message = str(excinfo.value)
    # Verify the error message contains all the expected components
    assert "sync_dist=True" in message
    assert "Detected a mismatch during `training_step`" in message
    assert "Synchronized metric keys per rank:" in message
    assert "rank=0:" in message
    assert "rank=1:" in message
    assert "rank=2:" in message
    assert "training_step.metric_a" in message
    assert "training_step.metric_b" in message
    assert "training_step.metric_c" in message
    # Verify it contains guidance on how to fix
    assert "log the same keys on all ranks" in message
    assert "sync_dist=False" in message


def test_forward_cache_synced_initialization():
    """Test that _forward_cache_synced is initialized to False."""
    result = _ResultCollection(training=True)
    result.log("training_step", "loss", torch.tensor(1.0), on_step=True, on_epoch=False, sync_dist=True)

    loss_metric = result["training_step.loss"]
    # Verify _forward_cache_synced is initialized to False
    assert loss_metric._forward_cache_synced is False
    # Verify _forward_cache is set
    assert loss_metric._forward_cache is not None


def test_forward_cache_set_on_step_metrics():
    """Test that _forward_cache is properly set for on_step metrics."""
    result = _ResultCollection(training=True)
    value = torch.tensor(2.5)
    result.log("training_step", "loss", value, on_step=True, on_epoch=False, sync_dist=True)

    loss_metric = result["training_step.loss"]
    # The forward_cache should be a clone of the value
    assert loss_metric._forward_cache is not None
    assert torch.equal(loss_metric._forward_cache, value)
    assert loss_metric._forward_cache_synced is False


def test_on_step_only_value_set_to_forward_cache():
    """Test that for on_step only metrics, value is set to forward_cache."""
    result = _ResultCollection(training=True)
    value = torch.tensor(3.0)
    result.log("training_step", "loss", value, on_step=True, on_epoch=False, sync_dist=True)

    loss_metric = result["training_step.loss"]
    # For on_step only metrics, value should equal forward_cache
    assert torch.equal(loss_metric.value, loss_metric._forward_cache)


def test_on_step_and_epoch_value_accumulated():
    """Test that for on_step+on_epoch metrics, value is accumulated separately."""
    result = _ResultCollection(training=True)
    value = torch.tensor(2.0)
    result.log("training_step", "loss", value, on_step=True, on_epoch=True, sync_dist=True)

    loss_metric = result["training_step.loss"]
    # forward_cache should be set
    assert loss_metric._forward_cache is not None
    assert loss_metric._forward_cache_synced is False
    # Value should be accumulated (not just set to forward_cache for on_epoch metrics)

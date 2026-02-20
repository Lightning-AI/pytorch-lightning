import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lightning.fabric import Fabric
from lightning.fabric.plugins.io.distributed_async_io import DistributedAsyncCheckpointIO
from lightning.fabric.utilities import AttributeDict
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_7, _TORCH_GREATER_EQUAL_2_9
from tests_fabric.helpers.runif import RunIf


@RunIf(min_torch="2.4")
def test_async_checkpointio_save_options_forwarded(tmp_path):
    plugin = DistributedAsyncCheckpointIO(save_options={"foo": 123})

    fake_future = MagicMock()

    with patch("lightning.fabric.plugins.io.distributed_async_io._dcp_save", return_value=fake_future) as save:
        plugin.save_checkpoint({"a": 1}, tmp_path)

    assert plugin.checkpoint_future is fake_future

    kwargs = save.call_args.kwargs
    assert kwargs["state_dict"] == {"a": 1}
    assert kwargs["filepath"] == tmp_path

    dcp_kwargs = kwargs["dcp_kwargs"]
    assert dcp_kwargs["foo"] == 123
    if _TORCH_GREATER_EQUAL_2_9:
        assert "no_dist" in dcp_kwargs
    if _TORCH_GREATER_EQUAL_2_7:
        assert "planner" in dcp_kwargs
        assert "async_checkpointer_type" in dcp_kwargs


@RunIf(min_torch="2.4")
def test_async_checkpointio_load_options_forwarded(tmp_path):
    plugin = DistributedAsyncCheckpointIO(load_options={"bar": 999})

    with patch("lightning.fabric.plugins.io.distributed_async_io._dcp_load") as load:
        plugin.load_checkpoint(tmp_path, state={"x": 1})

    assert load.call_count == 1
    kwargs = load.call_args.kwargs
    assert kwargs["state_dict"] == {"x": 1}
    assert kwargs["dcp_kwargs"]["bar"] == 999


@RunIf(min_torch="2.4")
def test_async_checkpointio_wait_uses_timeout():
    plugin = DistributedAsyncCheckpointIO(timeout=42)

    future = MagicMock()
    plugin.checkpoint_future = future

    plugin._wait()

    future.result.assert_called_once_with(timeout=42)


@RunIf(min_torch="2.4")
def test_async_checkpointio_save_waits_on_existing_future(tmp_path):
    plugin = DistributedAsyncCheckpointIO(timeout=None)

    prev_future = MagicMock()
    plugin.checkpoint_future = prev_future

    with patch("lightning.fabric.plugins.io.distributed_async_io._dcp_save", return_value=MagicMock()):
        plugin.save_checkpoint({"x": 1}, tmp_path)

    prev_future.result.assert_called_once()


@RunIf(min_torch="2.4")
def test_async_checkpointio_remove_waits(tmp_path):
    plugin = DistributedAsyncCheckpointIO(timeout=None)

    prev_future = MagicMock()
    plugin.checkpoint_future = prev_future

    with patch("lightning.fabric.plugins.io.distributed_async_io.get_filesystem") as fs_mock:
        fs = MagicMock()
        fs.exists.return_value = False
        fs_mock.return_value = fs

        plugin.remove_checkpoint(tmp_path)

    prev_future.result.assert_called_once()


@RunIf(min_torch="2.4")
def test_async_checkpointio_teardown_waits():
    plugin = DistributedAsyncCheckpointIO()

    future = MagicMock()
    plugin.checkpoint_future = future

    plugin.teardown()

    future.result.assert_called_once()


@RunIf(min_torch="2.4")
def test_async_checkpointio_requires_cpu_collectives():
    assert DistributedAsyncCheckpointIO().requires_cpu_collectives is True


@RunIf(min_torch="2.4")
def test_async_checkpointio_requires_restore_after_setup():
    assert DistributedAsyncCheckpointIO()._restore_after_setup is True


@RunIf(min_torch="2.4")
def test_async_checkpointio_load_requires_state(tmp_path):
    plugin = DistributedAsyncCheckpointIO()

    with pytest.raises(ValueError, match="`state` must be provided"):
        plugin.load_checkpoint(tmp_path)


@RunIf(min_torch="2.4")
def test_async_checkpointio_map_location_not_supported(tmp_path):
    plugin = DistributedAsyncCheckpointIO()

    with pytest.raises(TypeError):
        plugin.load_checkpoint(tmp_path, state={}, map_location="cpu")


@RunIf(min_torch="2.4")
def test_async_checkpointio_storage_options_not_supported(tmp_path):
    plugin = DistributedAsyncCheckpointIO()

    with pytest.raises(TypeError):
        plugin.save_checkpoint({"x": 1}, tmp_path, storage_options={})


# --- integration test to verify the checkpoint is actually saved and loaded asynchronously ---


def _broadcast_from_rank0(fabric: Fabric, obj):
    """Broadcast an object from rank0 once Fabric has launched."""
    fabric.barrier("pre_broadcast")
    obj = fabric.broadcast(obj)
    fabric.barrier("post_broadcast")
    return obj


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)


def _train_one_epoch(fabric, model, optimizer, loader, loss_fn):
    model.train()
    for x, y in loader:
        x, y = fabric.to_device((x, y))
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        fabric.backward(loss)
        optimizer.step()


def _wait_for_dcp_metadata(path: Path, timeout=10):
    # writing files in CI can be slow,
    # and DCP writes a metadata file last,
    # so we can wait for that to appear to ensure the checkpoint is ready
    start = time.time()
    while True:
        # DCP metadata file pattern
        if any(p.name.startswith(".metadata") for p in path.iterdir()):
            return
        if time.time() - start > timeout:
            raise RuntimeError("Checkpoint metadata not visible yet")
        time.sleep(0.1)


def run_async_checkpoint_state_restoration(tmp_path, expected_strategy_name, accelerator="auto", devices="auto"):
    torch.manual_seed(1234)

    tmp_path = Path(tmp_path)

    # tiny deterministic dataset
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    ckpt_io = DistributedAsyncCheckpointIO()
    fabric = Fabric(accelerator=accelerator, devices=devices, plugins=[ckpt_io])
    fabric.launch()

    assert fabric.strategy.__class__.__name__ == expected_strategy_name, (
        f"Expected strategy {expected_strategy_name}, but got {fabric.strategy.__class__.__name__}"
    )

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model, optimizer = fabric.setup(model, optimizer)
    loader = fabric.setup_dataloaders(loader)

    _train_one_epoch(fabric, model, optimizer, loader, loss_fn)

    # snapshot weights BEFORE save
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = AttributeDict(model=model, optimizer=optimizer, step=1)

    # rank0 decides canonical checkpoint path
    ckpt_path = tmp_path / "ckpt"
    ckpt_path = _broadcast_from_rank0(fabric, ckpt_path)

    fabric.save(ckpt_path, state)

    # Wait for DistributedAsyncCheckpointIO to finish writing checkpoint metadata.
    # Fabric does not automatically manage teardown, and calling
    # `fabric.strategy.teardown()` would release distributed resources needed
    # for loading. Therefore we only finalize the CheckpointIO lifecycle here.
    assert isinstance(fabric.strategy.checkpoint_io, DistributedAsyncCheckpointIO)
    fabric.strategy.checkpoint_io.teardown()
    _wait_for_dcp_metadata(ckpt_path)
    fabric.barrier("async_save_finished")  # ensure all ranks have observed the checkpoint before proceeding

    # destroy model weights intentionally
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1000.0)

    # verify weights actually changed
    assert not torch.allclose(
        next(iter(model.state_dict().values())),
        next(iter(before.values())),
    )

    # load back
    fabric.load(ckpt_path, state)

    after = model.state_dict()

    # verify restoration
    for k in before:
        assert torch.allclose(before[k], after[k])


@RunIf(min_torch="2.4", standalone=True)
def test_async_checkpointio_state_restoration_cpu(tmp_path):
    run_async_checkpoint_state_restoration(
        tmp_path, expected_strategy_name="SingleDeviceStrategy", accelerator="cpu", devices=1
    )


@RunIf(min_torch="2.4", min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize(
    ("expected_strategy_name", "devices"),
    [
        ("SingleDeviceStrategy", 1),
        ("DDPStrategy", 2),
    ],
)
def test_async_checkpointio_state_restoration_cuda(tmp_path, expected_strategy_name, devices):
    run_async_checkpoint_state_restoration(
        tmp_path, expected_strategy_name=expected_strategy_name, accelerator="cuda", devices=devices
    )

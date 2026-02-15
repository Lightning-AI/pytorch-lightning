from unittest.mock import MagicMock, patch

import pytest

from lightning.fabric.plugins.io.async_checkpoint_io import AsyncCheckpointIO


def test_async_checkpointio_save_options_forwarded(tmp_path):
    plugin = AsyncCheckpointIO(save_options={"foo": 123})

    fake_future = MagicMock()

    with patch("lightning.fabric.plugins.io.async_checkpoint_io._dcp_save", return_value=fake_future) as save:
        plugin.save_checkpoint({"a": 1}, tmp_path)

    assert plugin.checkpoint_future is fake_future

    kwargs = save.call_args.kwargs
    assert kwargs["state_dict"] == {"a": 1}
    assert kwargs["filepath"] == tmp_path

    dcp_kwargs = kwargs["dcp_kwargs"]
    assert dcp_kwargs["foo"] == 123
    assert "planner" in dcp_kwargs
    assert "async_checkpointer_type" in dcp_kwargs
    assert "no_dist" in dcp_kwargs


def test_async_checkpointio_load_options_forwarded(tmp_path):
    plugin = AsyncCheckpointIO(load_options={"bar": 999})

    with patch("lightning.fabric.plugins.io.async_checkpoint_io._dcp_load") as load:
        plugin.load_checkpoint(tmp_path, state={"x": 1})

    assert load.call_count == 1
    kwargs = load.call_args.kwargs
    assert kwargs["state_dict"] == {"x": 1}
    assert kwargs["dcp_kwargs"]["bar"] == 999


def test_async_checkpointio_wait_uses_timeout():
    plugin = AsyncCheckpointIO(timeout=42)

    future = MagicMock()
    plugin.checkpoint_future = future

    plugin._wait()

    future.result.assert_called_once_with(timeout=42)


def test_async_checkpointio_save_waits_on_existing_future(tmp_path):
    plugin = AsyncCheckpointIO(timeout=None)

    prev_future = MagicMock()
    plugin.checkpoint_future = prev_future

    with patch("lightning.fabric.plugins.io.async_checkpoint_io._dcp_save", return_value=MagicMock()):
        plugin.save_checkpoint({"x": 1}, tmp_path)

    prev_future.result.assert_called_once()


def test_async_checkpointio_remove_waits(tmp_path):
    plugin = AsyncCheckpointIO(timeout=None)

    prev_future = MagicMock()
    plugin.checkpoint_future = prev_future

    with patch("lightning.fabric.plugins.io.async_checkpoint_io.get_filesystem") as fs_mock:
        fs = MagicMock()
        fs.exists.return_value = False
        fs_mock.return_value = fs

        plugin.remove_checkpoint(tmp_path)

    prev_future.result.assert_called_once()


def test_async_checkpointio_teardown_waits():
    plugin = AsyncCheckpointIO(timeout=None)

    future = MagicMock()
    plugin.checkpoint_future = future

    plugin.teardown()

    future.result.assert_called_once()


def test_async_checkpointio_requires_cpu_collectives():
    assert AsyncCheckpointIO().requires_cpu_collectives is True


def test_async_checkpointio_requires_state_conversion():
    assert AsyncCheckpointIO()._requires_state_conversion is True


def test_async_checkpointio_load_requires_state(tmp_path):
    plugin = AsyncCheckpointIO()

    with pytest.raises(ValueError, match="`state` must be provided"):
        plugin.load_checkpoint(tmp_path)


def test_async_checkpointio_map_location_not_supported(tmp_path):
    plugin = AsyncCheckpointIO()

    with pytest.raises(TypeError):
        plugin.load_checkpoint(tmp_path, state={}, map_location="cpu")


def test_async_checkpointio_storage_options_not_supported(tmp_path):
    plugin = AsyncCheckpointIO()

    with pytest.raises(TypeError):
        plugin.save_checkpoint({"x": 1}, tmp_path, storage_options={})


# --- integration test to verify the checkpoint is actually saved and loaded asynchronously ---

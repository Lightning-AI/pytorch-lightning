import time
from threading import Event
from typing import Any, Optional
from unittest.mock import Mock

import pytest
import torch

from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.pytorch.plugins.io.async_plugin import AsyncCheckpointIO


class _CaptureCheckpointIO(CheckpointIO):
    def __init__(self) -> None:
        self.saved: Optional[dict[str, Any]] = None

    def save_checkpoint(self, checkpoint: dict[str, Any], path: str, storage_options: Optional[Any] = None) -> None:
        # Simulate some delay to increase race window
        time.sleep(0.05)
        # Store the received checkpoint object (not a deep copy) to inspect tensor values
        self.saved = checkpoint

    def load_checkpoint(self, path: str, map_location: Optional[Any] = None) -> dict[str, Any]:
        raise NotImplementedError

    def remove_checkpoint(self, path: str) -> None:
        pass


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_async_checkpoint_should_snapshot_values_before_mutation():
    base = _CaptureCheckpointIO()
    async_io = AsyncCheckpointIO(checkpoint_io=base)

    # a tensor that we will mutate after scheduling the save
    t = torch.tensor([0.0])
    ckpt = {"w": t}

    # schedule async save
    async_io.save_checkpoint(ckpt, path="unused")

    # mutate immediately afterward to mimic training thread stepping params
    t.add_(1.0)

    # ensure background thread finished
    async_io.teardown()

    assert base.saved is not None, "Async save did not run"

    # EXPECTATION: AsyncCheckpointIO should have captured value 0.0 (pre-mutation)
    # CURRENT BEHAVIOR (bug): it captures 1.0 because the dict holds references
    assert torch.allclose(base.saved["w"], torch.tensor([0.0])), (
        "AsyncCheckpointIO must snapshot the checkpoint (clone tensors) on the main thread "
        "to avoid races with parameter mutation; got mutated value instead"
    )


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("save_again", [False, True])
def test_async_checkpoint_remove_after_save(tmp_path, queued, save_again):
    started = Event()
    release = Event()
    blocked_path = tmp_path / "blocked.ckpt"
    removed_path = tmp_path / "queued.ckpt" if queued else blocked_path

    class BlockingCheckpointIO(TorchCheckpointIO):
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            if path == blocked_path and checkpoint["version"] == 1:
                started.set()
                assert release.wait(timeout=10)
            super().save_checkpoint(checkpoint, path, storage_options)

    checkpoint_io = AsyncCheckpointIO(BlockingCheckpointIO())
    try:
        checkpoint_io.save_checkpoint({"version": 1}, blocked_path)
        assert started.wait(timeout=10)
        if queued:
            checkpoint_io.save_checkpoint({"version": 1}, removed_path)
        checkpoint_io.remove_checkpoint(path=removed_path)
        if save_again:
            checkpoint_io.save_checkpoint({"version": 2}, removed_path)
    finally:
        release.set()
        checkpoint_io.teardown()

    if save_again:
        assert torch.load(removed_path, weights_only=True) == {"version": 2}
    else:
        assert not removed_path.exists()
    if queued:
        assert blocked_path.exists()
    assert checkpoint_io._executor is None


@pytest.mark.parametrize("operation", ["save", "remove"])
def test_async_checkpoint_operation_error(operation):
    error = OSError("checkpoint operation failed")
    base = TorchCheckpointIO()
    checkpoint_io = AsyncCheckpointIO(base)
    if operation == "save":
        base.save_checkpoint = Mock(side_effect=error)
    else:
        base.save_checkpoint = Mock()
        checkpoint_io.save_checkpoint({}, "unused")
        base.remove_checkpoint = Mock(side_effect=error)

    with pytest.raises(OSError, match="checkpoint operation failed"):
        try:
            if operation == "save":
                checkpoint_io.save_checkpoint({}, "unused")
            else:
                checkpoint_io.remove_checkpoint("unused")
        finally:
            checkpoint_io.teardown()

    assert checkpoint_io._error is error
    assert checkpoint_io._executor is None


@pytest.mark.parametrize("save_first", [False, True])
def test_async_checkpoint_remove_without_executor(tmp_path, save_first):
    path = tmp_path / "checkpoint.ckpt"
    base = TorchCheckpointIO()
    base.save_checkpoint({}, path)
    checkpoint_io = AsyncCheckpointIO(base)
    if save_first:
        checkpoint_io.save_checkpoint({}, path)
        checkpoint_io.teardown()

    try:
        checkpoint_io.remove_checkpoint(path)
        assert not path.exists()
        assert checkpoint_io._executor is None

        base.remove_checkpoint = Mock(side_effect=OSError("removal failed"))
        with pytest.raises(OSError, match="removal failed"):
            checkpoint_io.remove_checkpoint(path)
        assert checkpoint_io._executor is None
    finally:
        checkpoint_io.teardown()

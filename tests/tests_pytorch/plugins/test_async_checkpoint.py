import time
from threading import Event
from typing import Any, Optional

import numpy as np
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


class _MutableCheckpointState:
    def __init__(self) -> None:
        self.values = [1]


@pytest.mark.parametrize("keyword", [False, True])
def test_async_checkpoint_snapshots_mutable_state(tmp_path, keyword):
    started = Event()
    release = Event()

    class BlockingCheckpointIO(TorchCheckpointIO):
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            started.set()
            assert release.wait(timeout=10)
            super().save_checkpoint(checkpoint, path, storage_options)

    checkpoint_io = AsyncCheckpointIO(BlockingCheckpointIO())
    tensor = torch.tensor([1.0], requires_grad=True) * 2
    array = np.array([3.0])
    state = _MutableCheckpointState()
    checkpoint = {"weights": tensor, "callbacks": {"array": array, "state": state}, "same_weights": tensor}
    path = tmp_path / "snapshot.ckpt"
    try:
        if keyword:
            checkpoint_io.save_checkpoint(checkpoint=checkpoint, path=path)
        else:
            checkpoint_io.save_checkpoint(checkpoint, path)
        assert started.wait(timeout=10)
        with torch.no_grad():
            tensor.add_(10)
        array += 10
        state.values.append(2)
    finally:
        release.set()
        checkpoint_io.teardown()

    saved = torch.load(path, weights_only=False)
    torch.testing.assert_close(saved["weights"], torch.tensor([2.0]))
    np.testing.assert_array_equal(saved["callbacks"]["array"], [3.0])
    assert saved["callbacks"]["state"].values == [1]
    assert saved["same_weights"] is saved["weights"]
    assert not saved["weights"].requires_grad

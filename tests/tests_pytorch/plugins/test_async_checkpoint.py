import time
from typing import Any, Optional

import pytest
import torch

from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_async_checkpoint_clones_tensors_to_cpu():
    """Verify that _clone_tensor produces a CPU snapshot that does not share storage."""
    from lightning.pytorch.plugins.io.async_plugin import _clone_tensor

    t = torch.tensor([1.0, 2.0, 3.0])
    cloned = _clone_tensor(t)

    # cloned tensor should be on CPU
    assert cloned.device == torch.device("cpu"), f"Expected CPU tensor, got {cloned.device}"
    # values should match
    assert torch.equal(cloned, t)
    # cloned tensor should not share storage with the original
    assert cloned.data_ptr() != t.data_ptr()
    # mutation of the original must not affect the clone
    t.add_(1.0)
    assert torch.equal(cloned, torch.tensor([1.0, 2.0, 3.0]))

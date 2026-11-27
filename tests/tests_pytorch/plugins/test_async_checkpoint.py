import threading
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


def test_async_checkpoint_clears_error_after_raising():
    """Ensure that a background save error stored in ``_error`` is cleared once raised in ``save_checkpoint`` or
    ``teardown``, so subsequent ``teardown`` or ``save_checkpoint`` calls do not re-raise stale exceptions."""
    fail_next = True
    allow_worker_to_proceed = threading.Event()
    saved_paths: list[str] = []

    class _FlakyCheckpointIO(CheckpointIO):
        def save_checkpoint(self, checkpoint: dict[str, Any], path: str, storage_options: Optional[Any] = None) -> None:
            allow_worker_to_proceed.wait(timeout=5.0)
            if fail_next:
                raise RuntimeError("disk full")
            saved_paths.append(path)

        def load_checkpoint(self, path: str, map_location: Optional[Any] = None) -> dict[str, Any]:
            raise NotImplementedError

        def remove_checkpoint(self, path: str) -> None:
            pass

    async_io = AsyncCheckpointIO(checkpoint_io=_FlakyCheckpointIO())

    # 1. Error raised during teardown() should clear _error so subsequent saves/teardowns succeed.
    allow_worker_to_proceed.clear()
    async_io.save_checkpoint({"w": torch.tensor([1.0])}, path="fail_1.ckpt")
    allow_worker_to_proceed.set()
    with pytest.raises(RuntimeError, match="disk full"):
        async_io.teardown()
    assert async_io._error is None

    # Subsequent teardown should be a clean no-op
    async_io.teardown()

    # Subsequent save + teardown after recovery should succeed
    fail_next = False
    async_io.save_checkpoint({"w": torch.tensor([2.0])}, path="ok_1.ckpt")
    async_io.teardown()
    assert saved_paths == ["ok_1.ckpt"]

    # 2. Error raised on a subsequent save_checkpoint() should clear _error before dispatching a new task,
    # so teardown() does not re-raise even if the failure condition persists.
    fail_next = True
    allow_worker_to_proceed.clear()
    async_io.save_checkpoint({"w": torch.tensor([3.0])}, path="fail_2.ckpt")
    assert async_io._executor is not None
    allow_worker_to_proceed.set()
    # Wait for the background worker to finish and record _error without shutting down the executor
    async_io._executor.submit(lambda: None).result()
    assert async_io._error is not None

    with pytest.raises(RuntimeError, match="disk full"):
        async_io.save_checkpoint({"w": torch.tensor([4.0])}, path="unscheduled.ckpt")
    assert async_io._error is None

    # teardown() should not re-raise because unscheduled.ckpt was never dispatched
    async_io.teardown()
    assert saved_paths == ["ok_1.ckpt"]

    # Subsequent save + teardown after recovery should succeed
    fail_next = False
    async_io.save_checkpoint({"w": torch.tensor([5.0])}, path="ok_2.ckpt")
    async_io.teardown()
    assert saved_paths == ["ok_1.ckpt", "ok_2.ckpt"]


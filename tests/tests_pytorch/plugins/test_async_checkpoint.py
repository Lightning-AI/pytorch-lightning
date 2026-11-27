import time
from threading import Event
from typing import Any, Optional
from unittest.mock import Mock

import pytest
import torch

from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.pytorch.plugins.io.async_plugin import AsyncCheckpointIO
from tests_pytorch.helpers.runif import RunIf


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


class _BlockingTorchCheckpointIO(TorchCheckpointIO):
    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.saved = None

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        self.saved = checkpoint
        self.started.set()
        if not self.release.wait(timeout=30):
            raise TimeoutError("Checkpoint writer was not released")
        super().save_checkpoint(checkpoint, path, storage_options=storage_options)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=RunIf(min_cuda_gpus=1))])
@pytest.mark.parametrize("keyword", [False, True])
def test_async_checkpoint_stages_independent_cpu_tensors(tmp_path, device, keyword):
    base = _BlockingTorchCheckpointIO()
    async_io = AsyncCheckpointIO(checkpoint_io=base)
    weight = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4).requires_grad_()
    step = torch.tensor(7, device=device)
    checkpoint = {"state_dict": {"weight": weight[:, ::2]}, "optimizer_states": [{"step": step}], "epoch": 3}
    path = tmp_path / "snapshot.ckpt"
    try:
        if keyword:
            async_io.save_checkpoint(checkpoint=checkpoint, path=path)
        else:
            async_io.save_checkpoint(checkpoint, path)
        assert base.started.wait(timeout=30)
        assert base.saved is not None
        snapshot = base.saved["state_dict"]["weight"]
        assert snapshot.device.type == "cpu"
        assert base.saved["optimizer_states"][0]["step"].device.type == "cpu"
        assert not snapshot.requires_grad
        assert snapshot.grad_fn is None
        with torch.no_grad():
            weight.fill_(-1)
            step.add_(1)
    finally:
        base.release.set()
        async_io.teardown()

    loaded = base.load_checkpoint(path, weights_only=True)
    torch.testing.assert_close(
        loaded["state_dict"]["weight"], torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, ::2]
    )
    assert loaded["optimizer_states"][0]["step"].item() == 7
    assert loaded["epoch"] == 3


@RunIf(min_cuda_gpus=1)
def test_async_checkpoint_queued_saves_do_not_clone_cuda_storage(tmp_path):
    base = _BlockingTorchCheckpointIO()
    async_io = AsyncCheckpointIO(checkpoint_io=base)
    weight = torch.zeros(4 * 1024 * 1024, device="cuda")
    paths = [tmp_path / f"snapshot-{index}.ckpt" for index in range(2)]
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        async_io.save_checkpoint({"weight": weight}, paths[0])
        assert base.started.wait(timeout=30)
        weight.add_(1)
        async_io.save_checkpoint({"weight": weight}, paths[1])
        weight.add_(1)
        assert torch.cuda.max_memory_allocated() == allocated
    finally:
        base.release.set()
        async_io.teardown()

    for index, path in enumerate(paths):
        loaded = base.load_checkpoint(path, weights_only=True)
        torch.testing.assert_close(loaded["weight"], torch.full((weight.numel(),), float(index)))


def test_async_checkpoint_preserves_meta_tensors(tmp_path):
    base = TorchCheckpointIO()
    async_io = AsyncCheckpointIO(checkpoint_io=base)
    tensor = torch.empty((2, 3), device="meta", requires_grad=True)
    path = tmp_path / "meta.ckpt"
    try:
        async_io.save_checkpoint({"weight": tensor}, path)
    finally:
        async_io.teardown()

    loaded = base.load_checkpoint(path, weights_only=True)["weight"]
    assert loaded.device.type == "meta"
    assert loaded.shape == tensor.shape
    assert loaded.dtype == tensor.dtype
    assert not loaded.requires_grad


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

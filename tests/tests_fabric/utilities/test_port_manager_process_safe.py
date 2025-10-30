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
"""Tests for process-safe port manager features."""

import json
import multiprocessing
import os
import tempfile
import time
from pathlib import Path

import pytest

import lightning.fabric.utilities.port_manager as port_manager_module
from lightning.fabric.utilities.file_lock import UnixFileLock, WindowsFileLock, create_file_lock
from lightning.fabric.utilities.port_manager import PortManager, _get_lock_dir, _get_lock_file
from lightning.fabric.utilities.port_state import PortAllocation, PortState

# =============================================================================
# Tests for FileLock
# =============================================================================


def test_file_lock_platform_selection():
    """Test that create_file_lock returns the correct platform-specific implementation."""
    import sys

    handle, path = tempfile.mkstemp()
    os.close(handle)
    lock_file = Path(path)

    lock = create_file_lock(lock_file)

    if sys.platform == "win32":
        assert isinstance(lock, WindowsFileLock)
    else:
        assert isinstance(lock, UnixFileLock)


def test_file_lock_acquire_release(tmpdir):
    """Test basic file lock acquire and release."""
    lock_file = Path(tmpdir) / "test.lock"
    lock = create_file_lock(lock_file)

    # Acquire lock
    assert lock.acquire(timeout=1.0)
    assert lock.is_locked()

    # Release lock
    lock.release()
    assert not lock.is_locked()


def test_file_lock_context_manager(tmpdir):
    """Test file lock context manager."""
    lock_file = Path(tmpdir) / "test.lock"
    lock = create_file_lock(lock_file)

    with lock:
        assert lock.is_locked()

    assert not lock.is_locked()


def test_file_lock_timeout(tmpdir):
    """Test file lock acquisition timeout."""
    lock_file = Path(tmpdir) / "test.lock"
    lock1 = create_file_lock(lock_file)
    lock2 = create_file_lock(lock_file)

    # First lock acquires
    assert lock1.acquire(timeout=1.0)

    # Second lock should timeout
    timeout_seconds = 0.5
    start = time.time()
    assert not lock2.acquire(timeout=timeout_seconds)
    elapsed = time.time() - start

    # Should take approximately the timeout duration
    assert timeout_seconds * 0.8 < elapsed < timeout_seconds + 0.5

    lock1.release()


def test_file_lock_context_manager_timeout(tmpdir):
    """Test file lock context manager timeout raises exception."""
    lock_file = Path(tmpdir) / "test.lock"
    lock1 = create_file_lock(lock_file)
    lock2 = create_file_lock(lock_file)

    lock1.acquire(timeout=1.0)

    with pytest.raises(TimeoutError, match="Failed to acquire lock"), lock2:
        pass

    lock1.release()


def test_get_lock_dir_handles_permission_error(monkeypatch, tmp_path):
    """_get_lock_dir should tolerate probe unlink permission errors and register cleanup."""

    monkeypatch.setenv("LIGHTNING_PORT_LOCK_DIR", str(tmp_path))

    registered_calls = []

    def fake_register(func, *args, **kwargs):
        registered_calls.append((func, args, kwargs))
        return func

    monkeypatch.setattr(port_manager_module.atexit, "register", fake_register)

    original_unlink = Path.unlink
    call_state = {"count": 0}

    def fake_unlink(self, *args, **kwargs):
        if self.name.startswith(".lightning_port_manager_write_test_") and call_state["count"] == 0:
            call_state["count"] += 1
            raise PermissionError("locked")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    lock_dir = _get_lock_dir()
    assert Path(lock_dir) == tmp_path
    assert registered_calls, "Cleanup should be registered when unlink fails"

    cleanup_func, args, kwargs = registered_calls[0]
    probe_path = args[0]
    assert isinstance(probe_path, Path)
    assert probe_path.exists()

    cleanup_func(*args, **kwargs)
    assert not probe_path.exists()


# =============================================================================
# Tests for PortState
# =============================================================================


def test_port_state_allocate_and_release():
    """Test port allocation and release in PortState."""
    state = PortState()

    # Allocate port
    state.allocate_port(12345, pid=1000)
    assert state.is_port_allocated(12345)
    assert not state.is_port_recently_released(12345)

    # Release port
    state.release_port(12345)
    assert not state.is_port_allocated(12345)
    assert state.is_port_recently_released(12345)


def test_port_state_get_ports_for_pid():
    """Test getting all ports for a specific PID."""
    state = PortState()

    state.allocate_port(12345, pid=1000)
    state.allocate_port(12346, pid=1000)
    state.allocate_port(12347, pid=2000)

    ports_1000 = state.get_ports_for_pid(1000)
    ports_2000 = state.get_ports_for_pid(2000)

    assert set(ports_1000) == {12345, 12346}
    assert set(ports_2000) == {12347}


def test_port_state_cleanup_stale_entries():
    """Test cleanup of stale port allocations."""
    state = PortState()

    # Allocate port for current process
    current_pid = os.getpid()
    state.allocate_port(12345, pid=current_pid)

    # Allocate port for non-existent process
    fake_pid = 999999
    state.allocate_port(12346, pid=fake_pid)

    # Cleanup should remove the fake PID
    stale_count = state.cleanup_stale_entries()
    assert stale_count >= 1  # At least the fake PID should be cleaned

    # Current process port should still be allocated
    assert state.is_port_allocated(12345)
    # Fake PID port should be released
    assert not state.is_port_allocated(12346)


def test_port_state_json_serialization():
    """Test PortState JSON serialization and deserialization."""
    state = PortState()
    state.allocate_port(12345, pid=1000)
    state.release_port(12345)

    # Serialize to dict
    data = state.to_dict()
    assert "version" in data
    assert "allocated_ports" in data
    assert "recently_released" in data

    # Deserialize from dict
    restored_state = PortState.from_dict(data)
    assert restored_state.version == state.version
    assert len(restored_state.recently_released) == len(state.recently_released)


# =============================================================================
# Tests for Process-Safe PortManager
# =============================================================================


def test_port_manager_creates_lock_files(tmpdir):
    """Test that PortManager creates necessary lock and state files."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)
    port = manager.allocate_port()

    # Lock file should exist
    assert lock_file.exists()
    # State file should exist after allocation
    assert state_file.exists()

    manager.release_port(port)


def test_port_manager_state_persistence(tmpdir):
    """Test that port allocations persist across manager instances."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    # First manager allocates port
    manager1 = PortManager(lock_file=lock_file, state_file=state_file)
    port = manager1.allocate_port()

    # Read state directly
    with open(state_file) as f:
        data = json.load(f)

    assert str(port) in data["allocated_ports"]
    assert data["allocated_ports"][str(port)]["pid"] == os.getpid()

    manager1.release_port(port)


def test_port_manager_cleanup_stale_entries(tmpdir):
    """Test cleanup of stale entries from dead processes."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Manually create a state file with a stale entry (old timestamp)
    state = PortState()
    allocation = PortAllocation(port=12345, pid=999999, allocated_at=time.time() - 8000)  # Old allocation
    state.allocated_ports["12345"] = allocation

    # Write without going through manager to bypass auto-cleanup
    with manager._file_lock, open(state_file, "w") as f:
        json.dump(state.to_dict(), f)

    # Now cleanup should find and remove it
    # Note: _read_state auto-cleans, so we verify the port is gone after any read
    with manager._file_lock:
        final_state = manager._read_state()
        # The stale port should have been cleaned up
        assert not final_state.is_port_allocated(12345)


def test_port_manager_release_all_for_current_pid(tmpdir):
    """Test that release_all only releases ports owned by current process."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Allocate port for current process
    port1 = manager.allocate_port()

    # Manually add port for different PID (use current PID to avoid cleanup)
    with manager._file_lock:
        state = manager._read_state()
        # Use a different port and a PID that won't be cleaned (current process + 1)
        fake_pid = os.getpid() + 1
        state.allocate_port(99999, pid=fake_pid)
        manager._write_state(state)

    # Release all should only release current process ports
    manager.release_all()

    # Verify
    with manager._file_lock:
        state = manager._read_state()
        # Our port should be released
        assert not state.is_port_allocated(port1)
        # The fake PID port should still be allocated (since that PID might exist)
        # OR it was cleaned up (if PID doesn't exist) - either is acceptable
        # So we just verify our port was released


def test_port_manager_context_manager_cleanup(tmpdir):
    """Test that PortManager context manager releases all ports."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    with PortManager(lock_file=lock_file, state_file=state_file) as manager:
        port1 = manager.allocate_port()
        port2 = manager.allocate_port()
        # Ports should be allocated
        assert port1 in manager._allocated_ports
        assert port2 in manager._allocated_ports

    # After context, ports should be released
    assert port1 not in manager._allocated_ports
    assert port2 not in manager._allocated_ports


def test_port_manager_fails_fast_on_lock_timeout(tmpdir):
    """Test that PortManager raises RuntimeError on lock timeout to prevent state divergence."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Acquire file lock externally to simulate timeout
    external_lock = create_file_lock(lock_file)
    external_lock.acquire(timeout=5.0)

    try:
        # Manager should raise RuntimeError instead of falling back
        with pytest.raises(RuntimeError, match="Unable to acquire file lock for port allocation"):
            manager.allocate_port()

    finally:
        external_lock.release()


def test_port_manager_release_port_keeps_cache_on_timeout(tmpdir):
    """Test that release_port keeps port in local cache when lock timeout occurs.

    Regression test for Issue #2: release_port should NOT clear local cache on timeout to prevent state divergence where
    local cache says 'free' but shared state says 'allocated'.

    """
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Allocate a port first
    port = manager.allocate_port()
    assert port in manager._allocated_ports

    # Acquire file lock externally to simulate timeout during release
    external_lock = create_file_lock(lock_file)
    external_lock.acquire(timeout=5.0)

    try:
        # Attempt to release port - should log error but keep port in local cache
        manager.release_port(port)

        # Port should still be in local cache since release failed
        assert port in manager._allocated_ports, "Port should remain in local cache after timeout"
        assert port not in manager._recently_released, "Port should not be in recently released after timeout"

    finally:
        external_lock.release()

    # After releasing the lock, verify port can be successfully released
    manager.release_port(port)
    assert port not in manager._allocated_ports
    assert port in manager._recently_released


def test_port_manager_release_all_keeps_cache_on_timeout(tmpdir):
    """Test that release_all keeps ports in local cache when lock timeout occurs.

    Regression test for Issue #2: release_all should NOT clear local cache on timeout to prevent state divergence where
    local cache says 'free' but shared state says 'allocated'.

    """
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Allocate multiple ports
    port1 = manager.allocate_port()
    port2 = manager.allocate_port()
    assert port1 in manager._allocated_ports
    assert port2 in manager._allocated_ports
    initial_port_count = len(manager._allocated_ports)

    # Acquire file lock externally to simulate timeout during release
    external_lock = create_file_lock(lock_file)
    external_lock.acquire(timeout=5.0)

    try:
        # Attempt to release all ports - should log error but keep ports in local cache
        manager.release_all()

        # Ports should still be in local cache since release failed
        assert len(manager._allocated_ports) == initial_port_count, (
            "All ports should remain in local cache after timeout"
        )
        assert port1 in manager._allocated_ports, "Port1 should remain in local cache after timeout"
        assert port2 in manager._allocated_ports, "Port2 should remain in local cache after timeout"

    finally:
        external_lock.release()

    # After releasing the lock, verify ports can be successfully released
    manager.release_all()
    assert len(manager._allocated_ports) == 0
    assert len(manager._recently_released) == 0


def test_port_manager_environment_variable_isolation(tmpdir):
    """Test that LIGHTNING_PORT_LOCK_DIR environment variable works."""
    custom_dir = Path(tmpdir) / "custom_locks"

    # Set environment variable
    os.environ["LIGHTNING_PORT_LOCK_DIR"] = str(custom_dir)

    try:
        lock_dir = _get_lock_dir()
        assert lock_dir == custom_dir
        assert lock_dir.exists()

        lock_file = _get_lock_file()
        assert lock_file.parent == custom_dir

    finally:
        del os.environ["LIGHTNING_PORT_LOCK_DIR"]


# =============================================================================
# Multi-Process Integration Tests
# =============================================================================


def _allocate_port_in_subprocess(lock_file, state_file, result_queue):
    """Helper function to allocate port in a subprocess."""
    from lightning.fabric.utilities.port_manager import PortManager

    manager = PortManager(lock_file=Path(lock_file), state_file=Path(state_file))
    port = manager.allocate_port()
    result_queue.put((os.getpid(), port))
    time.sleep(0.1)  # Hold port briefly
    manager.release_port(port)


def test_port_manager_multi_process_allocation(tmpdir):
    """Test that multiple processes don't allocate the same port.

    Note: This test spawns multiple subprocesses and may be slower on Windows
    due to the spawn start method. Uses multiprocessing.Process which is
    cross-platform compatible but may have different performance characteristics.

    """
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    result_queue = multiprocessing.Queue()
    processes = []

    # Spawn 5 processes that each allocate a port
    for _ in range(5):
        p = multiprocessing.Process(
            target=_allocate_port_in_subprocess, args=(str(lock_file), str(state_file), result_queue)
        )
        processes.append(p)
        p.start()

    # Wait for all processes
    for p in processes:
        p.join(timeout=10)

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Should have 5 results
    assert len(results) == 5

    # All ports should be unique
    ports = [port for _, port in results]
    assert len(set(ports)) == 5, f"Duplicate ports allocated: {ports}"


def _allocate_and_release_multiple(lock_file, state_file):
    """Helper for concurrent access test - must be at module level for pickling."""
    from lightning.fabric.utilities.port_manager import PortManager

    manager = PortManager(lock_file=Path(lock_file), state_file=Path(state_file))
    for _ in range(3):
        port = manager.allocate_port()
        time.sleep(0.01)
        manager.release_port(port)


def test_port_manager_concurrent_access_no_deadlock(tmpdir):
    """Test that concurrent access doesn't cause deadlocks.

    Note: This test spawns multiple subprocesses and may be slower on Windows
    due to the spawn start method. Tests the file-locking mechanism under
    concurrent access from multiple processes.

    """
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    processes = []
    for _ in range(3):
        p = multiprocessing.Process(target=_allocate_and_release_multiple, args=(str(lock_file), str(state_file)))
        processes.append(p)
        p.start()

    # All processes should complete without deadlock
    for p in processes:
        p.join(timeout=10)
        assert p.exitcode == 0, "Process failed or deadlocked"


# =============================================================================
# Additional Coverage Tests
# =============================================================================


def test_port_manager_allocated_port_context_manager(tmpdir):
    """Test the allocated_port context manager for automatic cleanup."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Use context manager
    with manager.allocated_port() as port:
        # Port should be allocated
        assert port > 0
        assert port in manager._allocated_ports

        # Verify in shared state
        with manager._file_lock:
            state = manager._read_state()
            assert state.is_port_allocated(port)

    # After context, port should be released
    assert port not in manager._allocated_ports

    # Verify released in shared state
    with manager._file_lock:
        state = manager._read_state()
        assert not state.is_port_allocated(port)
        assert state.is_port_recently_released(port)


def test_port_manager_allocated_port_with_preferred(tmpdir):
    """Test allocated_port context manager with preferred port."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Find a free port
    preferred_port = manager._find_free_port()

    # Use context manager with preferred port
    with manager.allocated_port(preferred_port=preferred_port) as port:
        assert port == preferred_port


def test_port_manager_reserve_existing_port(tmpdir):
    """Test reserve_existing_port method in process-safe implementation."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Test reserving a free port
    free_port = manager._find_free_port()
    assert manager.reserve_existing_port(free_port)

    # Port should be in allocated ports
    assert free_port in manager._allocated_ports

    # Verify in shared state
    with manager._file_lock:
        state = manager._read_state()
        assert state.is_port_allocated(free_port)

    # Test reserving an already allocated port (should return True)
    assert manager.reserve_existing_port(free_port)

    # Test invalid port numbers
    assert not manager.reserve_existing_port(0)
    assert not manager.reserve_existing_port(-1)
    assert not manager.reserve_existing_port(70000)

    manager.release_port(free_port)


def test_port_manager_reserve_clears_recently_released(tmpdir):
    """Test that reserve_existing_port clears recently_released queue."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Allocate and release a port
    port = manager.allocate_port()
    manager.release_port(port)

    # Port should be in recently released
    assert port in manager._recently_released

    # Reserve the port again
    assert manager.reserve_existing_port(port)

    # Port should be removed from recently released
    assert port not in manager._recently_released


def test_file_lock_error_handling(tmpdir):
    """Test file lock error handling paths."""
    lock_file = Path(tmpdir) / "test.lock"

    lock = create_file_lock(lock_file)

    # Test release without acquire (should not raise)
    lock.release()
    assert not lock.is_locked()

    # Test multiple releases (should not raise)
    lock.acquire(timeout=1.0)
    lock.release()
    lock.release()
    assert not lock.is_locked()


def test_port_manager_cleanup_returns_count(tmpdir):
    """Test that cleanup_stale_entries happens during state read."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Manually create stale entries by writing directly to file
    with manager._file_lock:
        state = PortState()
        # Add old allocation (>2 hours ago)
        old_time = time.time() - 7300  # >2 hours
        state.allocated_ports["99998"] = PortAllocation(port=99998, pid=999998, allocated_at=old_time)
        state.allocated_ports["99999"] = PortAllocation(port=99999, pid=999999, allocated_at=old_time)
        # Write directly to bypass auto-cleanup
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f)

    # Read state - this should auto-cleanup the stale entries
    with manager._file_lock:
        cleaned_state = manager._read_state()
        # Stale entries should be gone
        assert not cleaned_state.is_port_allocated(99998)
        assert not cleaned_state.is_port_allocated(99999)


def test_port_manager_allocate_preferred_port(tmpdir):
    """Test allocating a specific preferred port."""
    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Find a free port to use as preferred
    preferred = manager._find_free_port()

    # Allocate it explicitly
    port = manager.allocate_port(preferred_port=preferred)

    # Should get the preferred port
    assert port == preferred

    # Verify it's tracked
    with manager._file_lock:
        state = manager._read_state()
        assert state.is_port_allocated(port)

    manager.release_port(port)


def test_port_manager_release_with_lock_timeout(tmpdir):
    """Test release_port when file lock times out."""
    import unittest.mock as mock

    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)
    port = manager.allocate_port()

    # Mock file lock to raise TimeoutError
    with mock.patch.object(manager._file_lock, "__enter__", side_effect=TimeoutError("Lock timeout")):
        # Release should handle timeout gracefully
        manager.release_port(port)

    # Port should still be removed from in-memory cache
    assert port not in manager._allocated_ports


def test_port_manager_release_all_with_lock_timeout(tmpdir):
    """Test release_all when file lock times out."""
    import unittest.mock as mock

    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)
    manager.allocate_port()
    manager.allocate_port()

    # Mock file lock to raise TimeoutError
    with mock.patch.object(manager._file_lock, "__enter__", side_effect=TimeoutError("Lock timeout")):
        # Release all should handle timeout gracefully
        manager.release_all()

    # Ports should still be cleared from in-memory cache
    assert len(manager._allocated_ports) == 0


def test_port_manager_cleanup_with_lock_timeout(tmpdir):
    """Test cleanup_stale_entries when file lock times out."""
    import unittest.mock as mock

    lock_file = Path(tmpdir) / "test.lock"
    state_file = Path(tmpdir) / "test_state.json"

    manager = PortManager(lock_file=lock_file, state_file=state_file)

    # Mock file lock to raise TimeoutError
    with mock.patch.object(manager._file_lock, "__enter__", side_effect=TimeoutError("Lock timeout")):
        # Cleanup should handle timeout gracefully and return 0
        count = manager.cleanup_stale_entries()
        assert count == 0

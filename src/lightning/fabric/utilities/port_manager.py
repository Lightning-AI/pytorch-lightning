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
"""Process-safe port allocation manager to prevent race conditions in distributed training."""

import atexit
import json
import logging
import os
import socket
import tempfile
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional

from lightning.fabric.utilities.file_lock import create_file_lock
from lightning.fabric.utilities.port_state import PortState

log = logging.getLogger(__name__)

# Size of the recently released ports queue
# This prevents immediate reuse of ports that were just released
# Set to 1024 to balance memory usage vs TIME_WAIT protection
_RECENTLY_RELEASED_PORTS_MAXLEN = 1024


def _get_lock_dir() -> Path:
    """Get directory for lock files, creating if needed.

    Uses LIGHTNING_PORT_LOCK_DIR environment variable if set, otherwise uses system temp directory.

    Returns:
        Path to lock directory

    Raises:
        RuntimeError: If directory is not writable

    """
    lock_dir = os.getenv("LIGHTNING_PORT_LOCK_DIR", tempfile.gettempdir())
    lock_path = Path(lock_dir)
    lock_path.mkdir(parents=True, exist_ok=True)

    # Validate directory is writable by creating and deleting a unique temp file
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".lightning_port_manager_write_test_", dir=lock_path)
    except (OSError, PermissionError) as e:
        raise RuntimeError(
            f"Port manager lock directory is not writable: {lock_path}. "
            f"Please ensure the directory exists and has write permissions, "
            f"or set LIGHTNING_PORT_LOCK_DIR to a writable location."
        ) from e

    test_file = Path(temp_name)
    with suppress(OSError):
        os.close(fd)

    with suppress(FileNotFoundError):
        try:
            test_file.unlink()
            return lock_path
        except PermissionError:
            log.debug("Port manager probe file could not be removed due to permission issues; scheduling cleanup")
        except OSError as e:
            log.debug(f"Port manager probe file removal failed with {e}; scheduling cleanup")

    atexit.register(_cleanup_probe_file, test_file)

    return lock_path


def _cleanup_probe_file(path: Path) -> None:
    """Best-effort removal of a temporary probe file at exit."""
    path.unlink(missing_ok=True)


def _get_lock_file() -> Path:
    """Get path to the port manager lock file.

    Returns:
        Path to lock file

    """
    return _get_lock_dir() / "lightning_port_manager.lock"


def _get_state_file() -> Path:
    """Get path to the port manager state file.

    Returns:
        Path to state file

    """
    return _get_lock_dir() / "lightning_port_manager_state.json"


class PortManager:
    """Process-safe port manager to prevent EADDRINUSE errors across multiple processes.

    This manager uses file-based locking to coordinate port allocation across multiple
    concurrent processes (e.g., pytest-xdist workers). It maintains shared state in a
    JSON file and uses platform-specific file locking for atomic operations.

    The manager maintains both thread-safety (for in-process coordination) and process-safety
    (for cross-process coordination), making it suitable for highly parallel test execution.

    Attributes:
        _lock: Thread-level lock for in-process synchronization
        _file_lock: File-level lock for cross-process synchronization
        _state_file: Path to shared state file
        _allocated_ports: In-memory cache of allocated ports
        _recently_released: In-memory cache of recently released ports

    """

    def __init__(self, lock_file: Optional[Path] = None, state_file: Optional[Path] = None) -> None:
        """Initialize the port manager.

        Args:
            lock_file: Optional path to lock file (defaults to system temp directory)
            state_file: Optional path to state file (defaults to system temp directory)

        """
        # Thread-level synchronization
        self._lock = threading.Lock()

        # File-based synchronization for process safety
        self._lock_file_path = lock_file or _get_lock_file()
        self._state_file = state_file or _get_state_file()
        self._file_lock = create_file_lock(self._lock_file_path)

        # In-memory cache for performance (process-local)
        self._allocated_ports: set[int] = set()
        self._recently_released: deque[int] = deque(maxlen=_RECENTLY_RELEASED_PORTS_MAXLEN)

        # Register cleanup to release all ports on exit
        atexit.register(self.release_all)

        log.debug(f"PortManager initialized with lock_dir={self._lock_file_path.parent}, pid={os.getpid()}")

    def _read_state(self) -> PortState:
        """Read state from file, cleaning stale entries.

        Low-level primitive - does NOT acquire lock.
        IMPORTANT: Caller must hold self._file_lock before calling.

        Returns:
            PortState instance with current state

        """
        if not self._state_file.exists():
            return PortState()  # Empty state

        try:
            with open(self._state_file) as f:
                data = json.load(f)
            state = PortState.from_dict(data)
            # Clean up stale entries on read
            state.cleanup_stale_entries()
            return state
        except (json.JSONDecodeError, OSError) as e:
            # Corrupted state, start fresh
            log.warning(f"Corrupted state file detected ({e}), starting with clean state")
            return PortState()

    def _write_state(self, state: PortState) -> None:
        """Atomically write state to file.

        Low-level primitive - does NOT acquire lock.
        IMPORTANT: Caller must hold self._file_lock before calling.

        Uses atomic write pattern: write to temp file, then rename.

        Args:
            state: PortState to write

        """
        temp_file = self._state_file.with_suffix(".tmp")

        try:
            # Ensure directory exists
            self._state_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file
            with open(temp_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic rename (platform-safe)
            temp_file.replace(self._state_file)
        except Exception as e:
            log.error(f"Failed to write state file: {e}")
            raise
        finally:
            # Clean up temp file if it still exists
            temp_file.unlink(missing_ok=True)

    def allocate_port(self, preferred_port: Optional[int] = None, max_attempts: int = 1000) -> int:
        """Allocate a free port with process-safe coordination.

        Args:
            preferred_port: If provided, try to allocate this specific port first
            max_attempts: Maximum number of attempts to find a free port

        Returns:
            An allocated port number

        Raises:
            RuntimeError: If unable to find a free port after max_attempts

        """
        with self._lock:  # Thread-safety
            try:
                with self._file_lock:  # Process-safety
                    state = self._read_state()
                    port = self._select_port(state, preferred_port, max_attempts)
                    state.allocate_port(port, pid=os.getpid())
                    self._write_state(state)

                    self._allocated_ports.add(port)

                    # Log diagnostics if queue utilization is high (>78%)
                    queue_count = len(state.recently_released)
                    threshold = int(_RECENTLY_RELEASED_PORTS_MAXLEN * 0.78)
                    if queue_count > threshold:
                        utilization_pct = (queue_count / _RECENTLY_RELEASED_PORTS_MAXLEN) * 100
                        log.warning(
                            f"Port queue utilization high: {queue_count} entries ({utilization_pct:.1f}% of capacity). "
                            f"Allocated port {port}. Active allocations: {len(state.allocated_ports)}"
                        )

                    log.debug(f"Allocated port {port} for pid={os.getpid()}")
                    return port

            except TimeoutError as e:
                log.error(
                    "Failed to acquire file lock for port allocation. "
                    "Remediation: (1) Retry the operation after a short delay, "
                    "(2) Check if another process is deadlocked holding the lock, "
                    "(3) Verify LIGHTNING_PORT_LOCK_DIR is accessible and not on a network filesystem."
                )
                raise RuntimeError(
                    "Unable to acquire file lock for port allocation. "
                    "This prevents process-safe coordination. "
                    "Check if another process is holding the lock or if the lock file is inaccessible."
                ) from e

        raise RuntimeError("Unexpected error allocating port")

    def _select_port(
        self,
        state: PortState,
        preferred_port: Optional[int],
        max_attempts: int,
    ) -> int:
        """Choose an available port based on preference and state."""
        if preferred_port is not None and self._is_port_available(preferred_port, state):
            return preferred_port

        for _ in range(max_attempts):
            candidate = self._find_free_port()
            if self._is_port_available(candidate, state):
                return candidate

        allocated_count = len(state.allocated_ports)
        queue_count = len(state.recently_released)
        raise RuntimeError(
            f"Failed to allocate a free port after {max_attempts} attempts. "
            f"Diagnostics: allocated={allocated_count}, recently_released={queue_count}"
        )

    def _is_port_available(self, port: int, state: PortState) -> bool:
        """Check if a port is available for allocation.

        Args:
            port: Port to check
            state: Current port state

        Returns:
            True if port is available

        """
        # Check if already allocated in shared state
        if state.is_port_allocated(port):
            return False

        # Check if recently released
        if state.is_port_recently_released(port):
            return False

        # Check if OS reports it as free
        return self._is_port_free(port)

    def release_port(self, port: int) -> None:
        """Release a previously allocated port with process-safe coordination.

        Args:
            port: Port number to release

        """
        with self._lock:  # Thread-safety
            release_succeeded = False
            try:
                with self._file_lock:  # Process-safety
                    state = self._read_state()
                    state.release_port(port)
                    self._write_state(state)
                    release_succeeded = True
            except TimeoutError:
                log.error(
                    f"Failed to acquire file lock when releasing port {port}. "
                    f"Port will remain allocated in shared state until process exits or stale cleanup (>2 hours). "
                    f"This may cause port exhaustion if it happens frequently. "
                    f"Keeping port in local cache to reflect true allocation state. "
                    f"Remediation: (1) Retry release_port() after a short delay, "
                    f"(2) Call cleanup_stale_entries() to force cleanup, "
                    f"(3) If deadlocked, restart affected processes."
                )

            # Only update in-memory cache if we successfully updated shared state
            # This prevents state divergence where local cache says "free" but shared state says "allocated"
            if release_succeeded and port in self._allocated_ports:
                self._allocated_ports.remove(port)
                self._recently_released.append(port)

    def release_all(self) -> None:
        """Release all ports allocated by this process."""
        with self._lock:  # Thread-safety
            release_succeeded = False
            try:
                with self._file_lock:  # Process-safety
                    state = self._read_state()
                    current_pid = os.getpid()

                    # Release ports owned by this PID
                    ports_to_release = state.get_ports_for_pid(current_pid)

                    for port in ports_to_release:
                        state.release_port(port)

                    if ports_to_release:
                        self._write_state(state)
                        log.debug(f"Released {len(ports_to_release)} port(s) for pid={current_pid}")

                    release_succeeded = True

            except TimeoutError:
                log.error(
                    "Failed to acquire file lock during release_all. "
                    "Ports will remain allocated in shared state until process exits or stale cleanup (>2 hours). "
                    "This may cause port exhaustion if it happens frequently. "
                    "Keeping ports in local cache to reflect true allocation state. "
                    "Remediation: (1) Retry release_all() after a short delay, "
                    "(2) Call cleanup_stale_entries() to force cleanup, "
                    "(3) If deadlocked, restart affected processes."
                )

            # Only clear in-memory cache if we successfully updated shared state
            # This prevents state divergence where local cache says "free" but shared state says "allocated"
            if release_succeeded:
                self._allocated_ports.clear()
                self._recently_released.clear()

    def cleanup_stale_entries(self) -> int:
        """Clean up stale port allocations from dead processes.

        Returns:
            Number of stale entries cleaned up

        """
        with self._lock:  # Thread-safety
            try:
                with self._file_lock:  # Process-safety
                    state = self._read_state()
                    stale_count = state.cleanup_stale_entries()

                    if stale_count > 0:
                        self._write_state(state)
                        log.info(f"Cleaned up {stale_count} stale port(s) from previous runs")

                    return stale_count

            except TimeoutError:
                log.warning(
                    "Failed to acquire file lock during cleanup. "
                    "Stale entries were not cleaned up. "
                    "Remediation: (1) Retry cleanup_stale_entries() after a short delay, "
                    "(2) Cleanup will occur automatically on next successful operation, "
                    "(3) Check for deadlocked processes holding the lock."
                )
                return 0

    def reserve_existing_port(self, port: int) -> bool:
        """Reserve a port that was allocated externally.

        Args:
            port: The externally assigned port to reserve.

        Returns:
            True if the port was reserved (or already reserved), False if the port value is invalid.

        """
        if port <= 0 or port > 65535:
            return False

        with self._lock:
            try:
                with self._file_lock:
                    state = self._read_state()

                    # If already allocated, that's fine
                    if state.is_port_allocated(port):
                        # Update in-memory cache
                        self._allocated_ports.add(port)
                        return True

                    # Allocate it
                    state.allocate_port(port, pid=os.getpid())
                    self._write_state(state)

                    # Update in-memory cache
                    self._allocated_ports.add(port)
                    # Remove from recently released if present
                    if port in self._recently_released:
                        self._recently_released = deque(
                            (p for p in self._recently_released if p != port), maxlen=_RECENTLY_RELEASED_PORTS_MAXLEN
                        )

                    return True

            except TimeoutError:
                log.error(
                    f"Failed to acquire file lock when reserving port {port}. "
                    "Cannot guarantee process-safe reservation. Returning False. "
                    "Remediation: (1) Retry reserve_existing_port() after a short delay, "
                    "(2) Use allocate_port() instead to let the manager choose a safe port, "
                    "(3) Check for lock contention or deadlocks."
                )
                # Do NOT update in-memory cache or claim success - this would create state divergence
                return False

    @contextmanager
    def allocated_port(self, preferred_port: Optional[int] = None) -> Iterator[int]:
        """Context manager for automatic port cleanup.

        Usage:
            with manager.allocated_port() as port:
                # Use port here
                pass
            # Port automatically released

        Args:
            preferred_port: Optional preferred port number

        Yields:
            Allocated port number

        """
        port = self.allocate_port(preferred_port=preferred_port)
        try:
            yield port
        finally:
            self.release_port(port)

    def __enter__(self) -> "PortManager":
        """Enter context manager - returns self for manager-level usage.

        Usage:
            with get_port_manager() as manager:
                port1 = manager.allocate_port()
                port2 = manager.allocate_port()
                # ... use ports
            # All ports from this process automatically released

        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        """Exit context manager - cleanup ports from this process."""
        self.release_all()
        return False  # Don't suppress exceptions

    @staticmethod
    def _find_free_port() -> int:
        """Find a free port using OS allocation.

        Returns:
            A port number that was free at the time of checking

        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Don't use SO_REUSEADDR - we need to match the behavior of TCPStore
        # which binds without it, so ports in TIME_WAIT will be rejected
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    @staticmethod
    def _is_port_free(port: int) -> bool:
        """Check if a specific port is available.

        Args:
            port: Port number to check

        Returns:
            True if the port is free, False otherwise

        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Don't use SO_REUSEADDR - we need to match the behavior of TCPStore
            # which binds without it, so ports in TIME_WAIT will be rejected
            s.bind(("", port))
            s.close()
            return True
        except OSError:
            return False


# Global singleton instance
_port_manager: Optional[PortManager] = None
_port_manager_lock = threading.Lock()


def get_port_manager() -> PortManager:
    """Get or create the global port manager instance.

    Returns:
        The global PortManager singleton

    """
    global _port_manager
    if _port_manager is None:
        with _port_manager_lock:
            if _port_manager is None:
                _port_manager = PortManager()
    return _port_manager


def find_free_network_port() -> int:
    """Find and reserve a free network port using the global port manager.

    Returns:
        A port number that is reserved and free at the time of allocation.

    """

    if "MASTER_PORT" in os.environ:
        master_port_str = os.environ["MASTER_PORT"]
        try:
            existing_port = int(master_port_str)
        except ValueError:
            pass
        else:
            port_manager = get_port_manager()
            if port_manager.reserve_existing_port(existing_port):
                return existing_port

    port_manager = get_port_manager()
    return port_manager.allocate_port()

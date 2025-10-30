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
"""Platform-abstracted file locking for cross-process coordination."""

import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional

log = logging.getLogger(__name__)


class FileLock(ABC):
    """Abstract base class for platform-specific file locking.

    File locks enable process-safe coordination by providing exclusive access to shared resources across multiple
    processes. This abstract interface allows platform-specific implementations while maintaining a consistent API.

    """

    def __init__(self, lock_file: Path) -> None:
        """Initialize the file lock.

        Args:
            lock_file: Path to the lock file

        """
        self._lock_file = lock_file
        self._fd: Optional[int] = None
        self._is_locked = False

    @abstractmethod
    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire the lock, blocking up to timeout seconds.

        Args:
            timeout: Maximum seconds to wait for lock acquisition

        Returns:
            True if lock was acquired, False if timeout occurred

        """

    @abstractmethod
    def release(self) -> None:
        """Release the lock if held."""

    def is_locked(self) -> bool:
        """Check if this instance currently holds the lock.

        Returns:
            True if lock is currently held by this instance

        """
        return self._is_locked

    def __enter__(self) -> "FileLock":
        """Enter context manager - acquire lock."""
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock on {self._lock_file} within timeout")
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        """Exit context manager - release lock."""
        self.release()
        return False  # Don't suppress exceptions

    def __del__(self) -> None:
        """Cleanup - ensure lock is released and file descriptor closed."""
        if self._is_locked:
            with suppress(Exception):
                self.release()

        if self._fd is not None:
            with suppress(Exception):
                os.close(self._fd)


class UnixFileLock(FileLock):
    """File locking using fcntl.flock for Unix-like systems (Linux, macOS).

    Uses fcntl.flock() which provides advisory locking. This implementation uses LOCK_EX (exclusive lock) with LOCK_NB
    (non-blocking) for timeout support.

    """

    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire exclusive lock using fcntl.flock.

        Args:
            timeout: Maximum seconds to wait for lock

        Returns:
            True if lock acquired, False if timeout occurred

        """
        import fcntl

        # Ensure lock file exists and open it
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file.touch(exist_ok=True)

        if self._fd is None:
            self._fd = os.open(str(self._lock_file), os.O_RDWR | os.O_CREAT)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to acquire exclusive lock non-blockingly
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._is_locked = True
                return True
            except OSError:
                # Lock held by another process, wait and retry
                time.sleep(0.1)

        # Timeout - log warning
        elapsed = time.time() - start_time
        log.warning(f"Lock acquisition timeout after {elapsed:.1f}s for {self._lock_file}")
        return False

    def release(self) -> None:
        """Release the lock using fcntl.flock."""
        if not self._is_locked or self._fd is None:
            return

        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._is_locked = False
        except OSError as e:
            log.warning(f"Error releasing lock on {self._lock_file}: {e}")


class WindowsFileLock(FileLock):
    """File locking using msvcrt.locking for Windows systems.

    Uses msvcrt.locking() which provides mandatory locking on Windows. This implementation uses LK_NBLCK (non-blocking
    exclusive lock) for timeout support.

    """

    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire exclusive lock using msvcrt.locking.

        Args:
            timeout: Maximum seconds to wait for lock

        Returns:
            True if lock acquired, False if timeout occurred

        """
        import msvcrt

        # Ensure lock file exists and open it
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file.touch(exist_ok=True)

        if self._fd is None:
            self._fd = os.open(str(self._lock_file), os.O_RDWR | os.O_CREAT)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to lock 1 byte at file position 0
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                self._is_locked = True
                return True
            except OSError:
                # Lock held by another process, wait and retry
                time.sleep(0.1)

        # Timeout - log warning
        elapsed = time.time() - start_time
        log.warning(f"Lock acquisition timeout after {elapsed:.1f}s for {self._lock_file}")
        return False

    def release(self) -> None:
        """Release the lock using msvcrt.locking."""
        if not self._is_locked or self._fd is None:
            return

        import msvcrt

        try:
            # Unlock the byte we locked
            msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            self._is_locked = False
        except OSError as e:
            log.warning(f"Error releasing lock on {self._lock_file}: {e}")


def create_file_lock(lock_file: Path) -> FileLock:
    """Factory function to create platform-appropriate file lock.

    Args:
        lock_file: Path to the lock file

    Returns:
        Platform-specific FileLock instance

    """
    if sys.platform == "win32":
        return WindowsFileLock(lock_file)
    return UnixFileLock(lock_file)

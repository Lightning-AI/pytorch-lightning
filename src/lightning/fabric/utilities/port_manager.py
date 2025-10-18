#  Copyright The Lightning AI team.
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
"""Port allocation manager to prevent race conditions in distributed training."""

import atexit
import socket
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

# Maximum number of recently released ports to track before reuse
_RECENTLY_RELEASED_PORTS_MAXLEN = 256


class PortManager:
    """Thread-safe port manager to prevent EADDRINUSE errors.

    This manager maintains a global registry of allocated ports to ensure that multiple concurrent tests don't try to
    use the same port. While this doesn't completely eliminate the race condition with external processes, it prevents
    internal collisions within the test suite.

    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._allocated_ports: set[int] = set()
        # Recently released ports are kept in a queue to avoid immediate reuse
        self._recently_released: deque[int] = deque(maxlen=_RECENTLY_RELEASED_PORTS_MAXLEN)
        # Register cleanup to release all ports on exit
        atexit.register(self.release_all)

    def allocate_port(self, preferred_port: Optional[int] = None, max_attempts: int = 100) -> int:
        """Allocate a free port, ensuring it's not already reserved.

        Args:
            preferred_port: If provided, try to allocate this specific port first
            max_attempts: Maximum number of attempts to find a free port

        Returns:
            An allocated port number

        Raises:
            RuntimeError: If unable to find a free port after max_attempts

        """
        with self._lock:
            # If a preferred port is specified and available, use it
            if (
                preferred_port is not None
                and preferred_port not in self._allocated_ports
                and preferred_port not in self._recently_released
                and self._is_port_free(preferred_port)
            ):
                self._allocated_ports.add(preferred_port)
                return preferred_port

            # Try to find a free port
            for attempt in range(max_attempts):
                port = self._find_free_port()

                # Skip ports that were recently released to avoid TIME_WAIT conflicts
                if port in self._recently_released:
                    continue

                if port not in self._allocated_ports:
                    self._allocated_ports.add(port)
                    return port

            raise RuntimeError(
                f"Failed to allocate a free port after {max_attempts} attempts. "
                f"Currently allocated ports: {len(self._allocated_ports)}"
            )

    def release_port(self, port: int) -> None:
        """Release a previously allocated port.

        Args:
            port: Port number to release

        """
        with self._lock:
            if port in self._allocated_ports:
                self._allocated_ports.remove(port)
                # Add to the back of the queue; oldest will be evicted when queue is full
                self._recently_released.append(port)

    def release_all(self) -> None:
        """Release all allocated ports."""
        with self._lock:
            self._allocated_ports.clear()
            self._recently_released.clear()

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
            if port in self._allocated_ports:
                return True

            # Remove from recently released queue if present (we're explicitly reserving it)
            if port in self._recently_released:
                # Create a new deque without this port
                self._recently_released = deque(
                    (p for p in self._recently_released if p != port), maxlen=_RECENTLY_RELEASED_PORTS_MAXLEN
                )

            self._allocated_ports.add(port)
            return True

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

    @staticmethod
    def _find_free_port() -> int:
        """Find a free port using OS allocation.

        Returns:
            A port number that was free at the time of checking

        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

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
"""State management for process-safe port allocation."""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Maximum age for port allocations before considering them stale (2 hours)
_STALE_PORT_AGE_SECONDS = 7200

# Maximum age for recently released entries (2 hours)
_RECENTLY_RELEASED_MAX_AGE_SECONDS = 7200

# Maximum number of recently released entries to retain
_RECENTLY_RELEASED_MAX_LEN = 1024


@dataclass
class PortAllocation:
    """Information about an allocated port."""

    port: int
    pid: int
    allocated_at: float
    process_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation

        """
        return {
            "pid": self.pid,
            "allocated_at": self.allocated_at,
            "process_name": self.process_name,
        }

    @classmethod
    def from_dict(cls, port: int, data: dict[str, Any]) -> "PortAllocation":
        """Create from dictionary.

        Args:
            port: The port number
            data: Dictionary with allocation info

        Returns:
            PortAllocation instance

        """
        return cls(
            port=port,
            pid=data["pid"],
            allocated_at=data["allocated_at"],
            process_name=data.get("process_name", ""),
        )

    def is_stale(self, current_time: float) -> bool:
        """Check if this allocation is stale (too old).

        Args:
            current_time: Current timestamp

        Returns:
            True if allocation is stale

        """
        return (current_time - self.allocated_at) > _STALE_PORT_AGE_SECONDS


@dataclass
class RecentlyReleasedEntry:
    """Information about a recently released port."""

    port: int
    released_at: float
    pid: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation

        """
        return {
            "port": self.port,
            "released_at": self.released_at,
            "pid": self.pid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecentlyReleasedEntry":
        """Create from dictionary.

        Args:
            data: Dictionary with release info

        Returns:
            RecentlyReleasedEntry instance

        """
        return cls(
            port=data["port"],
            released_at=data["released_at"],
            pid=data["pid"],
        )

    def is_stale(self, current_time: float) -> bool:
        """Check if this entry is stale (too old).

        Args:
            current_time: Current timestamp

        Returns:
            True if entry is stale

        """
        return (current_time - self.released_at) > _RECENTLY_RELEASED_MAX_AGE_SECONDS


@dataclass
class PortState:
    """Shared state for port allocations across processes.

    This class represents the JSON-serializable state stored in the state file. It tracks allocated ports with ownership
    information and recently released ports.

    """

    version: str = "1.0"
    allocated_ports: dict[str, PortAllocation] = field(default_factory=dict)
    recently_released: list[RecentlyReleasedEntry] = field(default_factory=list)

    def allocate_port(self, port: int, pid: int) -> None:
        """Allocate a port for a specific process.

        Args:
            port: Port number to allocate
            pid: Process ID of the owner

        """
        allocation = PortAllocation(
            port=port,
            pid=pid,
            allocated_at=time.time(),
            process_name=_get_process_name(pid),
        )
        self.allocated_ports[str(port)] = allocation

    def release_port(self, port: int) -> None:
        """Release an allocated port.

        Args:
            port: Port number to release

        """
        port_str = str(port)
        if port_str in self.allocated_ports:
            allocation = self.allocated_ports[port_str]
            # Add to recently released
            entry = RecentlyReleasedEntry(
                port=port,
                released_at=time.time(),
                pid=allocation.pid,
            )
            self.recently_released.append(entry)
            self._trim_recently_released()
            # Remove from allocated
            del self.allocated_ports[port_str]

    def is_port_allocated(self, port: int) -> bool:
        """Check if a port is currently allocated.

        Args:
            port: Port number to check

        Returns:
            True if port is allocated

        """
        return str(port) in self.allocated_ports

    def is_port_recently_released(self, port: int) -> bool:
        """Check if a port was recently released.

        Args:
            port: Port number to check

        Returns:
            True if port is in recently released list

        """
        return any(entry.port == port for entry in self.recently_released)

    def cleanup_stale_entries(self) -> int:
        """Remove stale allocations and recently released entries.

        This includes:
        - Ports from dead processes
        - Ports allocated too long ago (>2 hours)
        - Recently released entries older than 2 hours

        Returns:
            Number of stale entries removed

        """
        current_time = time.time()
        stale_count = 0

        # Clean up stale allocated ports
        stale_ports = []
        for port_str, allocation in self.allocated_ports.items():
            if not _is_pid_alive(allocation.pid) or allocation.is_stale(current_time):
                stale_ports.append(port_str)
                stale_count += 1

        for port_str in stale_ports:
            port = int(port_str)
            # Capture allocation info before releasing (for logging)
            allocation = self.allocated_ports[port_str]
            pid = allocation.pid
            self.release_port(port)
            log.debug(f"Cleaned up stale port {port} from pid={pid}")

        # Clean up stale recently released entries
        original_count = len(self.recently_released)
        self.recently_released = [entry for entry in self.recently_released if not entry.is_stale(current_time)]
        if len(self.recently_released) > _RECENTLY_RELEASED_MAX_LEN:
            # Keep only the most recent entries if stale cleanup still exceeds max length
            self.recently_released = self.recently_released[-_RECENTLY_RELEASED_MAX_LEN:]
        stale_count += original_count - len(self.recently_released)

        return stale_count

    def get_ports_for_pid(self, pid: int) -> list[int]:
        """Get all ports allocated by a specific process.

        Args:
            pid: Process ID

        Returns:
            List of port numbers owned by this PID

        """
        return [int(port_str) for port_str, allocation in self.allocated_ports.items() if allocation.pid == pid]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation

        """
        return {
            "version": self.version,
            "allocated_ports": {port_str: alloc.to_dict() for port_str, alloc in self.allocated_ports.items()},
            "recently_released": [entry.to_dict() for entry in self.recently_released],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortState":
        """Create from dictionary.

        Args:
            data: Dictionary with state data

        Returns:
            PortState instance

        """
        allocated_ports = {
            port_str: PortAllocation.from_dict(int(port_str), alloc_data)
            for port_str, alloc_data in data.get("allocated_ports", {}).items()
        }

        recently_released = [
            RecentlyReleasedEntry.from_dict(entry_data) for entry_data in data.get("recently_released", [])
        ]

        state = cls(
            version=data.get("version", "1.0"),
            allocated_ports=allocated_ports,
            recently_released=recently_released,
        )

        state._trim_recently_released()
        return state

    def _trim_recently_released(self) -> None:
        """Ensure recently released queue stays within configured bound."""
        if len(self.recently_released) > _RECENTLY_RELEASED_MAX_LEN:
            excess = len(self.recently_released) - _RECENTLY_RELEASED_MAX_LEN
            # Remove the oldest entries (front of the list)
            self.recently_released = self.recently_released[excess:]


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with given PID is still running.

    Args:
        pid: Process ID to check

    Returns:
        True if process is alive

    """
    try:
        # Send signal 0 - doesn't actually send a signal, just checks if process exists
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _get_process_name(pid: int) -> str:
    """Get the name of a process by PID.

    Args:
        pid: Process ID

    Returns:
        Process name or empty string if not available

    """
    try:
        # Try to get process name using psutil if available
        import psutil

        process = psutil.Process(pid)
        return process.name()
    except (ImportError, Exception):
        # psutil not available or process lookup failed
        return ""

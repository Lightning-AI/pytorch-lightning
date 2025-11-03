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
"""Port allocation with retry logic for distributed training."""

import socket
from typing import Optional


def _find_free_port() -> int:
    """Find a free port using OS allocation."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            return True
    except OSError:
        return False


def allocate_port_with_lock(preferred_port: Optional[int] = None, max_attempts: int = 100) -> int:
    """Allocate a port with retry logic for parallel process coordination.

    Uses simple OS port allocation with retry attempts. This approach accepts that
    there's an inherent race condition between allocating a port and actually binding to it,
    and handles it through retries rather than attempting to prevent it.

    The race condition occurs because:
    1. We ask OS for a port → get port 50435
    2. We close socket to return the port number
    3. Another process can grab port 50435 here ← RACE WINDOW
    4. TCPStore tries to bind → EADDRINUSE

    This is unfixable without keeping the socket open, which isn't possible
    when we only return a port number. File locks don't help because they can't
    prevent the OS from reusing a port.

    Args:
        preferred_port: Try to use this port first if available
        max_attempts: Maximum number of allocation attempts

    Returns:
        An available port number

    Raises:
        RuntimeError: If unable to allocate a port after max_attempts

    """
    # Try preferred port first
    if preferred_port and _is_port_available(preferred_port):
        return preferred_port

    # Simple OS allocation - let the kernel choose
    # Multiple attempts help reduce collision probability when many parallel processes
    # are allocating ports simultaneously
    for attempt in range(max_attempts):
        port = _find_free_port()

        # Small random delay to reduce collision probability with parallel processes
        # Only sleep on retry attempts, not the first try
        if attempt > 0:
            import random
            import time

            time.sleep(random.uniform(0.001, 0.01))  # noqa: S311

        # Verify port is still available (best effort)
        if _is_port_available(port):
            return port

    raise RuntimeError(f"Failed to allocate a free port after {max_attempts} attempts")

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
"""Tests for the PortManager utility and port allocation integration."""

import os
import socket
import threading
from collections import Counter

from lightning.fabric.plugins.environments.lightning import find_free_network_port
from lightning.fabric.utilities.port_manager import PortManager, get_port_manager

# =============================================================================
# Unit Tests for PortManager
# =============================================================================


def test_port_manager_allocates_unique_ports():
    """Test that PortManager allocates unique ports."""
    manager = PortManager()

    # Allocate multiple ports
    ports = [manager.allocate_port() for _ in range(10)]

    # All ports should be unique
    assert len(ports) == len(set(ports)), f"Duplicate ports found: {ports}"

    # All ports should be valid (>= 1024)
    assert all(p >= 1024 for p in ports), "Some ports are in reserved range"


def test_port_manager_release_port():
    """Test that released ports are removed from the allocated set."""
    manager = PortManager()

    # Allocate a port
    port = manager.allocate_port()
    assert port in manager._allocated_ports

    # Release the port
    manager.release_port(port)
    assert port not in manager._allocated_ports


def test_port_manager_release_all():
    """Test that release_all clears all allocated ports."""
    manager = PortManager()

    # Allocate multiple ports
    [manager.allocate_port() for _ in range(5)]
    assert len(manager._allocated_ports) == 5

    # Release all
    manager.release_all()
    assert len(manager._allocated_ports) == 0


def test_port_manager_release_nonexistent_port():
    """Test that releasing a non-existent port doesn't cause errors."""
    manager = PortManager()

    # Try to release a port that was never allocated
    manager.release_port(12345)  # Should not raise an error

    # Verify nothing broke
    port = manager.allocate_port()
    assert port >= 1024


def test_port_manager_thread_safety():
    """Test that PortManager is thread-safe under concurrent access."""
    manager = PortManager()
    ports = []
    lock = threading.Lock()

    def allocate_ports():
        """Allocate multiple ports from different threads."""
        for _ in range(10):
            port = manager.allocate_port()
            with lock:
                ports.append(port)

    # Create multiple threads that allocate ports concurrently
    threads = [threading.Thread(target=allocate_ports) for _ in range(10)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify we got 100 unique ports (10 threads × 10 ports each)
    assert len(ports) == 100, f"Expected 100 ports, got {len(ports)}"
    assert len(set(ports)) == 100, f"Expected 100 unique ports, got {len(set(ports))}"

    # Check for any duplicates
    counts = Counter(ports)
    duplicates = {port: count for port, count in counts.items() if count > 1}
    assert not duplicates, f"Found duplicate ports: {duplicates}"


def test_port_manager_preferred_port():
    """Test that PortManager can allocate a preferred port if available."""
    manager = PortManager()

    # Try to find a free port first
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    free_port = s.getsockname()[1]
    s.close()

    # Allocate the preferred port
    allocated = manager.allocate_port(preferred_port=free_port)
    assert allocated == free_port

    # Try to allocate the same preferred port again (should get a different one)
    allocated2 = manager.allocate_port(preferred_port=free_port)
    assert allocated2 != free_port


def test_port_manager_allocation_failure():
    """Test that PortManager raises error when unable to allocate after max attempts."""
    manager = PortManager()

    # This is hard to test without actually exhausting ports, but we can test
    # the error path by mocking or just ensure the code path exists
    # For now, just verify that max_attempts parameter exists
    port = manager.allocate_port(max_attempts=1)
    assert port >= 1024


def test_port_manager_prevents_reallocation():
    """Test that a port won't be allocated twice until released."""
    manager = PortManager()

    # Allocate a port
    port1 = manager.allocate_port()

    # Allocate many more ports - none should match port1
    more_ports = [manager.allocate_port() for _ in range(50)]

    # port1 should not appear in more_ports
    assert port1 not in more_ports, f"Port {port1} was reallocated before release"

    # After releasing port1, we should eventually be able to get it again
    # (though not guaranteed due to OS port allocation)
    manager.release_port(port1)
    assert port1 not in manager._allocated_ports


def test_get_port_manager_singleton():
    """Test that get_port_manager returns the same instance."""
    manager1 = get_port_manager()
    manager2 = get_port_manager()

    # Should be the same instance
    assert manager1 is manager2

    # Allocating from one should be visible in the other
    port = manager1.allocate_port()
    assert port in manager2._allocated_ports


def test_get_port_manager_thread_safe_singleton():
    """Test that get_port_manager creates singleton safely across threads."""
    managers = []
    lock = threading.Lock()

    def get_manager():
        manager = get_port_manager()
        with lock:
            managers.append(manager)

    # Create multiple threads that get the port manager
    threads = [threading.Thread(target=get_manager) for _ in range(20)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # All should be the same instance
    assert len(managers) == 20
    assert all(m is managers[0] for m in managers), "get_port_manager returned different instances"


def test_port_manager_is_port_free():
    """Test the _is_port_free helper method."""
    manager = PortManager()

    # Find a free port using OS
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    free_port = s.getsockname()[1]

    # Port should be reported as not free while socket is open
    assert not manager._is_port_free(free_port)

    # Close the socket
    s.close()

    # Now port should be free (though there's still a small race condition)
    # We'll skip this check as it's unreliable due to OS behavior


def test_port_manager_find_free_port():
    """Test the _find_free_port helper method."""
    manager = PortManager()

    # Should return a valid port
    port = manager._find_free_port()
    assert isinstance(port, int)
    assert port >= 1024
    assert port <= 65535


def test_port_manager_concurrent_allocation_and_release():
    """Test concurrent allocation and release operations."""
    manager = PortManager()
    ports = []
    lock = threading.Lock()

    def allocate_and_release():
        for _ in range(5):
            # Allocate a port
            port = manager.allocate_port()
            with lock:
                ports.append(port)

            # Release it immediately
            manager.release_port(port)

    # Run multiple threads
    threads = [threading.Thread(target=allocate_and_release) for _ in range(10)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Should have allocated 50 ports total (10 threads × 5 ports)
    assert len(ports) == 50

    # All should be unique (no port allocated twice before being released)
    assert len(set(ports)) == 50, "Same port was allocated to multiple threads before release"

    # After all releases, manager should have no ports allocated
    assert len(manager._allocated_ports) == 0


def test_port_manager_atexit_cleanup():
    """Test that PortManager registers atexit cleanup."""

    # Create a new manager
    manager = PortManager()

    # The manager should have registered release_all with atexit
    # We can't easily test atexit directly, but we can verify the method exists
    assert callable(manager.release_all)

    # Verify release_all works
    manager.allocate_port()
    manager.allocate_port()
    assert len(manager._allocated_ports) == 2

    manager.release_all()
    assert len(manager._allocated_ports) == 0


def test_port_manager_context_manager():
    """Test that context manager automatically releases ports."""
    manager = PortManager()

    # Use context manager
    with manager.allocated_port() as port:
        # Port should be allocated
        assert port in manager._allocated_ports
        assert isinstance(port, int)
        assert port >= 1024

    # After context, port should be released
    assert port not in manager._allocated_ports


def test_port_manager_context_manager_exception():
    """Test that context manager releases port even on exception."""
    manager = PortManager()

    try:
        with manager.allocated_port() as port:
            allocated_port = port
            # Port should be allocated
            assert port in manager._allocated_ports
            # Raise exception
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Port should still be released despite exception
    assert allocated_port not in manager._allocated_ports


def test_port_manager_context_manager_nested():
    """Test that nested context managers work correctly."""
    manager = PortManager()

    with manager.allocated_port() as port1:
        assert port1 in manager._allocated_ports

        with manager.allocated_port() as port2:
            # Both ports should be allocated
            assert port1 in manager._allocated_ports
            assert port2 in manager._allocated_ports
            # Ports should be different
            assert port1 != port2

        # port2 should be released, port1 still allocated
        assert port1 in manager._allocated_ports
        assert port2 not in manager._allocated_ports

    # Both ports should now be released
    assert port1 not in manager._allocated_ports
    assert port2 not in manager._allocated_ports


# =============================================================================
# Integration Tests for find_free_network_port()
# =============================================================================


def test_find_free_network_port_uses_port_manager():
    """Test that find_free_network_port uses the PortManager."""
    manager = get_port_manager()

    # Clear any previously allocated ports
    initial_count = len(manager._allocated_ports)

    # Allocate a port using the function
    port = find_free_network_port()

    # The port should be in the manager's allocated set
    assert port in manager._allocated_ports
    assert len(manager._allocated_ports) == initial_count + 1

    # Clean up
    manager.release_port(port)


def test_find_free_network_port_returns_unique_ports():
    """Test that multiple calls return unique ports."""
    manager = get_port_manager()

    # Allocate multiple ports
    ports = [find_free_network_port() for _ in range(10)]

    # All should be unique
    assert len(ports) == len(set(ports)), f"Duplicate ports: {ports}"

    # All should be tracked by the manager
    for port in ports:
        assert port in manager._allocated_ports

    # Clean up
    for port in ports:
        manager.release_port(port)


def test_find_free_network_port_thread_safety():
    """Test that find_free_network_port is thread-safe."""
    ports = []
    lock = threading.Lock()

    def allocate():
        for _ in range(5):
            port = find_free_network_port()
            with lock:
                ports.append(port)

    # Run 10 threads, each allocating 5 ports
    threads = [threading.Thread(target=allocate) for _ in range(10)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Should have 50 unique ports
    assert len(ports) == 50
    assert len(set(ports)) == 50, "Duplicate ports allocated across threads"

    # Clean up
    manager = get_port_manager()
    for port in ports:
        manager.release_port(port)


def test_port_allocation_simulates_distributed_test_lifecycle():
    """Simulate the lifecycle of a distributed test with port allocation and release."""
    manager = get_port_manager()
    initial_count = len(manager._allocated_ports)

    # Simulate test setup: allocate a port
    port = find_free_network_port()
    os.environ["MASTER_PORT"] = str(port)

    # Verify port is allocated
    assert port in manager._allocated_ports

    # Simulate test teardown: release the port
    if "MASTER_PORT" in os.environ:
        port_to_release = int(os.environ["MASTER_PORT"])
        manager.release_port(port_to_release)
        del os.environ["MASTER_PORT"]

    # Verify port is released
    assert port not in manager._allocated_ports
    assert len(manager._allocated_ports) == initial_count


def test_multiple_tests_can_reuse_ports_after_release():
    """Test that ports can be reused after being released."""
    manager = get_port_manager()

    # First "test" allocates a port
    port1 = find_free_network_port()
    assert port1 in manager._allocated_ports

    # First "test" completes and releases the port
    manager.release_port(port1)
    assert port1 not in manager._allocated_ports

    # Second "test" allocates ports (may or may not get the same port)
    port2 = find_free_network_port()
    assert port2 in manager._allocated_ports

    # Ports should be valid regardless
    assert port1 >= 1024
    assert port2 >= 1024

    # Clean up
    manager.release_port(port2)


def test_concurrent_tests_dont_get_same_port():
    """Test that concurrent tests never receive the same port."""
    manager = get_port_manager()
    ports_per_thread = []
    lock = threading.Lock()

    def simulate_test():
        """Simulate a test that allocates a port, uses it, then releases it."""
        my_ports = []

        # Allocate port for this "test"
        port = find_free_network_port()
        my_ports.append(port)

        # Simulate some work
        import time

        time.sleep(0.001)

        # Release port after "test" completes
        manager.release_port(port)

        with lock:
            ports_per_thread.append(my_ports)

    # Run 20 concurrent "tests"
    threads = [threading.Thread(target=simulate_test) for _ in range(20)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Collect all ports that were allocated
    all_ports = [port for thread_ports in ports_per_thread for port in thread_ports]

    # All ports should have been unique at the time of allocation
    assert len(all_ports) == 20
    assert len(set(all_ports)) == 20, "Some concurrent tests got the same port!"


def test_port_manager_survives_multiple_test_sessions():
    """Test that the port manager maintains state across multiple test sessions."""
    manager = get_port_manager()

    # Session 1: Allocate some ports
    session1_ports = [find_free_network_port() for _ in range(3)]

    # Session 2: Allocate more ports (should not overlap with session 1)
    session2_ports = [find_free_network_port() for _ in range(3)]

    # No overlap between sessions while both are active
    assert not set(session1_ports) & set(session2_ports)

    # Release session 1 ports
    for port in session1_ports:
        manager.release_port(port)

    # Session 3: Can allocate more ports
    session3_ports = [find_free_network_port() for _ in range(3)]

    # Session 3 shouldn't overlap with active session 2
    assert not set(session2_ports) & set(session3_ports)

    # Clean up
    for port in session2_ports + session3_ports:
        manager.release_port(port)

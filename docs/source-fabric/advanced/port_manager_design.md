# Process-Safe Port Manager Design

**Author:** LittlebullGit
**Date:** October 2024
**Status:** Design Document
**Component:** `lightning.fabric.utilities.port_manager`

## Executive Summary

This document describes the design and implementation of a process-safe port allocation manager for PyTorch Lightning. The port manager prevents `EADDRINUSE` errors in distributed training tests by coordinating port allocation across multiple concurrent processes using file-based locking.

## Problem Statement

### Current Limitations

The original `PortManager` implementation is thread-safe but **not process-safe**:

1. **Thread-safe only:** Uses `threading.Lock()` which only protects within a single Python process
1. **In-memory state:** Port allocations stored in process-local memory (`set[int]`)
1. **Global singleton per process:** Each process has its own instance with no inter-process communication
1. **Race conditions in CI:** When GPU tests run in batches (e.g., 5 concurrent pytest workers), multiple processes may allocate the same port

### Failure Scenario

```
Process A (pytest-xdist worker 0):
  - Allocates port 12345
  - Stores in local memory

Process B (pytest-xdist worker 1):
  - Unaware of Process A's allocation
  - Allocates same port 12345
  - Stores in local memory

Both processes attempt to bind → EADDRINUSE error
```

### Requirements

1. **Process-safe:** Coordinate port allocation across multiple concurrent processes
1. **Platform-neutral:** Support Linux, macOS, and Windows
1. **Backward compatible:** Existing API must continue to work unchanged
1. **Test-focused:** Optimized for test suite usage (up to 1-hour ML training tests)
1. **Performance:** Minimal overhead (\<10ms per allocation)
1. **Robust cleanup:** Handle process crashes, stale locks, and orphaned ports
1. **Configurable:** Support isolated test runs via environment variables

## Architecture Overview

### Components

```
┌─────────────────────────────────────────────────────────┐
│                     PortManager                          │
│  - Public API (allocate_port, release_port)             │
│  - Context manager support (__enter__, __exit__)        │
│  - In-memory cache for performance                      │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
┌───────▼─────────┐         ┌────────▼────────┐
│   File Lock     │         │   State Store   │
│  (Platform      │         │   (JSON file)   │
│   specific)     │         └─────────────────┘
└─────────────────┘
        │
        ├─ UnixFileLock (fcntl.flock)
        └─ WindowsFileLock (msvcrt.locking)
```

### File-Based Coordination

**Lock File:** `lightning_port_manager.lock`

- Platform-specific file locking mechanism
- Ensures atomic read-modify-write operations
- 30-second acquisition timeout with deadlock detection

**State File:** `lightning_port_manager_state.json`

- JSON-formatted shared state
- Atomic writes (temp file + rename)
- PID-based port ownership tracking

**Default Location:** System temp directory (from `tempfile.gettempdir()`)

**Override:** Set `LIGHTNING_PORT_LOCK_DIR` environment variable

## Detailed Design

### 1. Platform Abstraction Layer

#### FileLock Interface

```python
class FileLock(ABC):
    """Abstract base class for platform-specific file locking."""

    @abstractmethod
    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire the lock, blocking up to timeout seconds.

        Args:
            timeout: Maximum seconds to wait for lock

        Returns:
            True if lock acquired, False on timeout
        """

    @abstractmethod
    def release(self) -> None:
        """Release the lock."""

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError("Failed to acquire lock")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
```

#### Unix Implementation (fcntl)

```python
class UnixFileLock(FileLock):
    """File locking using fcntl.flock (Linux, macOS)."""

    def acquire(self, timeout: float = 30.0) -> bool:
        import fcntl
        import time

        start = time.time()
        while time.time() - start < timeout:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (OSError, IOError):
                time.sleep(0.1)
        return False
```

#### Windows Implementation (msvcrt)

```python
class WindowsFileLock(FileLock):
    """File locking using msvcrt.locking (Windows)."""

    def acquire(self, timeout: float = 30.0) -> bool:
        import msvcrt
        import time

        start = time.time()
        while time.time() - start < timeout:
            try:
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                time.sleep(0.1)
        return False
```

### 2. State Management

#### State Schema

```json
{
  "version": "1.0",
  "allocated_ports": {
    "12345": {
      "pid": 54321,
      "allocated_at": 1729774800.123,
      "process_name": "pytest-xdist-worker-0"
    },
    "12346": {
      "pid": 54322,
      "allocated_at": 1729774801.456,
      "process_name": "pytest-xdist-worker-1"
    }
  },
  "recently_released": [
    {
      "port": 12340,
      "released_at": 1729774700.789,
      "pid": 54320
    }
  ]
}
```

#### State Operations

**Design Pattern: Low-Level Primitives**

Both `_read_state()` and `_write_state()` are low-level primitives that **do not manage locking**. The caller (high-level operations like `allocate_port()`, `release_port()`) is responsible for holding the file lock during the entire read-modify-write cycle. This ensures:

- **Atomicity:** Lock held across entire operation
- **Symmetry:** Both primitives follow same pattern
- **Clarity:** Clear separation between low-level and high-level operations

**Read State (Low-Level):**

```python
def _read_state(self) -> PortState:
    """Read state from file, cleaning stale entries.

    Low-level primitive - does NOT acquire lock.
    IMPORTANT: Caller must hold self._file_lock before calling.
    """
    if not self._state_file.exists():
        return PortState()  # Empty state

    try:
        with open(self._state_file, 'r') as f:
            data = json.load(f)
        state = PortState.from_dict(data)
        state.cleanup_stale_entries()  # Remove dead PIDs
        return state
    except (json.JSONDecodeError, OSError):
        # Corrupted state, start fresh
        log.warning("Corrupted state file detected, starting with clean state")
        return PortState()
```

**Write State (Low-Level):**

```python
def _write_state(self, state: PortState) -> None:
    """Atomically write state to file.

    Low-level primitive - does NOT acquire lock.
    IMPORTANT: Caller must hold self._file_lock before calling.

    Uses atomic write pattern: write to temp file, then rename.
    """
    temp_file = self._state_file.with_suffix('.tmp')

    try:
        with open(temp_file, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)

        # Atomic rename (platform-safe)
        temp_file.replace(self._state_file)
    finally:
        # Clean up temp file if it still exists
        temp_file.unlink(missing_ok=True)
```

**Runtime Safety Checks (Optional):**

To prevent misuse, we can add runtime assertions:

```python
def _read_state(self) -> PortState:
    """Read state from file."""
    if not self._file_lock.is_locked():
        raise RuntimeError("_read_state called without holding lock")
    # ... rest of implementation

def _write_state(self, state: PortState) -> None:
    """Write state to file."""
    if not self._file_lock.is_locked():
        raise RuntimeError("_write_state called without holding lock")
    # ... rest of implementation
```

**High-Level Operations (Manage Locking):**

High-level public methods manage the file lock for entire operations:

```python
def release_port(self, port: int) -> None:
    """Release a port (high-level operation).

    Manages locking internally - calls low-level primitives.
    """
    with self._file_lock:  # <-- Acquire lock
        state = self._read_state()       # Low-level read
        state.release_port(port)          # Modify state
        self._write_state(state)          # Low-level write
    # <-- Release lock

    # Update in-memory cache (outside lock)
    if port in self._allocated_ports:
        self._allocated_ports.remove(port)
        self._recently_released.append(port)
```

**Pattern Summary:**

```
Low-Level (_read_state, _write_state):
  - Do NOT acquire lock
  - Assume lock is held
  - Private methods (underscore prefix)
  - Called only by high-level operations

High-Level (allocate_port, release_port, cleanup_stale_entries):
  - Acquire lock using `with self._file_lock:`
  - Call low-level primitives inside critical section
  - Public API methods
  - Hold lock for entire read-modify-write cycle
```

### 3. Port Allocation Algorithm

```python
def allocate_port(self, preferred_port: Optional[int] = None,
                  max_attempts: int = 1000) -> int:
    """Allocate a free port with process-safe coordination.

    Algorithm:
    1. Acquire file lock
    2. Read current state from file
    3. Clean up stale entries (dead PIDs, old timestamps)
    4. Check if preferred port is available
    5. Otherwise, find free port via OS
    6. Verify port not in allocated or recently_released
    7. Add to allocated_ports with current PID
    8. Write updated state to file
    9. Release file lock
    10. Update in-memory cache
    """

    with self._file_lock:
        state = self._read_state()

        # Try preferred port
        if preferred_port and self._is_port_available(preferred_port, state):
            port = preferred_port
        else:
            # Find free port
            for _ in range(max_attempts):
                port = self._find_free_port()
                if self._is_port_available(port, state):
                    break
            else:
                raise RuntimeError(f"Failed to allocate port after {max_attempts} attempts")

        # Allocate in state
        state.allocate_port(port, pid=os.getpid())
        self._write_state(state)

        # Update in-memory cache
        self._allocated_ports.add(port)

        return port
```

### 4. Cleanup Strategy

#### Three-Tier Cleanup

**1. Normal Cleanup (atexit)**

```python
def release_all(self) -> None:
    """Release all ports allocated by this process."""
    with self._file_lock:
        state = self._read_state()
        current_pid = os.getpid()

        # Release ports owned by this PID
        ports_to_release = [
            port for port, info in state.allocated_ports.items()
            if info['pid'] == current_pid
        ]

        for port in ports_to_release:
            state.release_port(port)

        self._write_state(state)
```

**2. Stale Entry Cleanup**

```python
def cleanup_stale_entries(self) -> int:
    """Remove ports from dead processes."""
    with self._file_lock:
        state = self._read_state()

        stale_count = 0
        for port, info in list(state.allocated_ports.items()):
            if not self._is_pid_alive(info['pid']):
                state.release_port(port)
                stale_count += 1

        # Remove old recently_released entries (>2 hours)
        cutoff = time.time() - 7200  # 2 hours
        state.recently_released = [
            entry for entry in state.recently_released
            if entry['released_at'] > cutoff
        ]

        self._write_state(state)
        return stale_count
```

**3. Time-Based Cleanup**

- Ports allocated >2 hours ago are considered stale
- Automatically cleaned on next allocation
- Prevents leaked ports from hung tests

### 5. Context Manager Support

```python
class PortManager:
    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup ports from this process."""
        self.release_all()
        return False  # Don't suppress exceptions
```

**Usage Patterns:**

```python
# Pattern 1: Explicit management (backward compatible)
manager = get_port_manager()
port = manager.allocate_port()
try:
    # ... use port
finally:
    manager.release_port(port)

# Pattern 2: Single port context manager (existing)
with get_port_manager().allocated_port() as port:
    # ... use port

# Pattern 3: Manager-level context manager (NEW)
with get_port_manager() as manager:
    port1 = manager.allocate_port()
    port2 = manager.allocate_port()
    # ... use ports
# Automatic cleanup
```

### 6. Configuration

#### Environment Variables

**LIGHTNING_PORT_LOCK_DIR**

- Override default lock file location
- Default: `tempfile.gettempdir()`
- Use case: Isolate parallel CI jobs

```bash
# Example: Parallel CI jobs on same machine
export LIGHTNING_PORT_LOCK_DIR=/tmp/lightning_ci_job_1
pytest tests/

# Job 2
export LIGHTNING_PORT_LOCK_DIR=/tmp/lightning_ci_job_2
pytest tests/
```

#### File Paths

```python
def _get_lock_dir() -> Path:
    """Get directory for lock files, creating if needed."""
    lock_dir = os.getenv("LIGHTNING_PORT_LOCK_DIR", tempfile.gettempdir())
    lock_path = Path(lock_dir)
    lock_path.mkdir(parents=True, exist_ok=True)
    return lock_path

def _get_lock_file() -> Path:
    return _get_lock_dir() / "lightning_port_manager.lock"

def _get_state_file() -> Path:
    return _get_lock_dir() / "lightning_port_manager_state.json"
```

### 7. Pytest Integration

#### Session Hooks

```python
# In tests/tests_fabric/conftest.py and tests/tests_pytorch/conftest.py

def pytest_sessionstart(session):
    """Clean stale port state at session start."""
    from lightning.fabric.utilities.port_manager import get_port_manager

    manager = get_port_manager()
    stale_count = manager.cleanup_stale_entries()

    if stale_count > 0:
        print(f"Cleaned up {stale_count} stale port(s) from previous runs")

def pytest_sessionfinish(session, exitstatus):
    """Final cleanup at session end."""
    from lightning.fabric.utilities.port_manager import get_port_manager

    manager = get_port_manager()
    manager.cleanup_stale_entries()
```

#### Test-Level Cleanup (Enhanced)

Existing retry logic in `pytest_runtest_makereport` is enhanced to:

1. Release ports before retry
1. Clean up stale entries
1. Wait for OS TIME_WAIT state

## Performance Considerations

### Optimization Strategies

**1. In-Memory Cache**

- Keep process-local cache of allocated ports
- Only consult file state on allocation/release
- Reduces file I/O by ~90%

**2. Lazy Cleanup**

- Stale entry cleanup on allocation, not on every read
- Batch cleanup operations
- Amortize cleanup cost

**3. Lock Minimization**

- Hold file lock only during critical section
- Release immediately after state write
- Typical lock hold time: \<5ms

**4. Non-Blocking Fast Path**

- Try non-blocking lock first
- Fall back to blocking with timeout
- Reduces contention in common case

### Performance Targets

| Operation       | Target | Notes                     |
| --------------- | ------ | ------------------------- |
| Port allocation | \<10ms | Including file lock + I/O |
| Port release    | \<5ms  | Simple state update       |
| Stale cleanup   | \<50ms | May scan 100+ entries     |
| Lock contention | \<1%   | Processes rarely overlap  |

## Error Handling

### Lock Acquisition Failure

```python
try:
    with self._file_lock:
        # ... allocate port
except TimeoutError as e:
    # Fail fast to prevent state divergence
    log.error("Failed to acquire file lock for port allocation")
    raise RuntimeError(
        "Unable to acquire file lock for port allocation. "
        "This prevents process-safe coordination. "
        "Check if another process is holding the lock or if the lock file is inaccessible."
    ) from e
```

**Rationale**: We fail fast on lock timeout instead of falling back to OS allocation. Fallback would bypass the shared state, allowing multiple processes to allocate the same port, defeating the purpose of process-safe coordination. By raising an error, we force the caller to handle the exceptional case explicitly rather than silently accepting a race condition.

### Corrupted State File

```python
try:
    state = json.load(f)
except json.JSONDecodeError:
    log.warning("Corrupted state file, starting fresh")
    return PortState()  # Empty state
```

### Dead PID Detection

```python
def _is_pid_alive(self, pid: int) -> bool:
    """Check if process is still running."""
    try:
        os.kill(pid, 0)  # Signal 0 = existence check
        return True
    except (OSError, ProcessLookupError):
        return False
```

## Security Considerations

### File Permissions

- Lock and state files created with default umask
- No sensitive data stored (only port numbers and PIDs)
- Consider restrictive permissions in multi-user environments

### Race Conditions

- **Time-of-check-to-time-of-use:** Mitigated by holding lock during entire allocation
- **Stale lock detection:** Verify PID before breaking lock
- **Atomic writes:** Use temp file + rename pattern

## Testing Strategy

### Unit Tests

1. **File locking:** Test acquire/release on each platform
1. **State serialization:** JSON encode/decode
1. **PID validation:** Alive/dead detection
1. **Stale cleanup:** Remove dead process ports
1. **Context manager:** Enter/exit behavior

### Integration Tests

1. **Multi-process allocation:** Spawn 5+ processes, verify unique ports
1. **Process crash recovery:** Kill process mid-allocation, verify cleanup
1. **Lock timeout:** Simulate deadlock, verify recovery
1. **Stress test:** 1000+ allocations across processes

### Platform-Specific Tests

- Run full suite on Linux, macOS, Windows
- Verify file locking behavior on each platform
- Test in CI with pytest-xdist `-n 5`

### Rollback Plan

If critical issues arise:

1. Revert the commit that introduced process-safe port manager (all changes are self-contained in the new files)
1. Remove any leftover lock/state files from the temp directory: `rm -f /tmp/lightning_port_manager*` or the custom `LIGHTNING_PORT_LOCK_DIR` location
1. The implementation maintains full backward compatibility - all existing tests pass without modification

**Note**: State files are self-contained JSON files with no schema migrations required. Stale entries will be automatically cleaned up on next session start.

## Monitoring and Metrics

### Logging Events

**DEBUG Level:**

- Port allocation/release
- Lock acquisition

**WARNING Level:**

- Lock contention (wait >1s)
- Stale lock detection
- Corrupted state recovery
- High queue utilization (>80%)

**ERROR Level:**

- Lock timeout
- File I/O failures
- Allocation failures after max retries

### Example Log Output

```
DEBUG: PortManager initialized with lock_dir=/tmp, pid=12345
DEBUG: Allocated port 12345 for pid=12345 in 3.2ms
WARNING: Lock contention detected, waited 1.5s for acquisition
WARNING: Cleaned up 3 stale ports from dead processes
ERROR: Failed to allocate port after 1000 attempts (allocated=50, queue=1020/1024)
```

## Future Enhancements

### Possible Improvements

1. **Port pool pre-allocation:** Reserve block of ports upfront
1. **Distributed coordination:** Support multi-machine coordination (Redis/etcd)
1. **Port affinity:** Prefer certain port ranges per process
1. **Metrics collection:** Track allocation patterns, contention rates
1. **Web UI:** Visualize port allocation state (debug tool)

### Not Planned

- Cross-network coordination (out of scope)
- Port forwarding/tunneling (different concern)
- Permanent port reservations (tests only)

## Appendix

### A. File Format Examples

**Empty State:**

```json
{
  "version": "1.0",
  "allocated_ports": {},
  "recently_released": []
}
```

**Active Allocations:**

```json
{
  "version": "1.0",
  "allocated_ports": {
    "12345": {
      "pid": 54321,
      "allocated_at": 1729774800.123,
      "process_name": "pytest-xdist-worker-0"
    }
  },
  "recently_released": [
    {
      "port": 12340,
      "released_at": 1729774700.789,
      "pid": 54320
    }
  ]
}
```

### B. Platform Compatibility

| Platform | Lock Mechanism | Tested Versions     |
| -------- | -------------- | ------------------- |
| Linux    | fcntl.flock    | Ubuntu 20.04, 22.04 |
| macOS    | fcntl.flock    | macOS 13, 14        |
| Windows  | msvcrt.locking | Windows Server 2022 |

### C. References

- [fcntl documentation](https://docs.python.org/3/library/fcntl.html)
- [msvcrt documentation](https://docs.python.org/3/library/msvcrt.html)
- [pytest-xdist](https://github.com/pytest-dev/pytest-xdist)
- [EADDRINUSE explanation](https://man7.org/linux/man-pages/man2/bind.2.html)

### D. FAQ

**Q: Why file-based instead of shared memory?**
A: File-based is more portable, survives process crashes better, and works well with pytest-xdist's process model.

**Q: What happens if the state file is deleted mid-run?**
A: Next allocation will create a fresh state file. Some ports may be double-allocated until processes resync, but retry logic will recover.

**Q: How do I debug port allocation issues?**
A: Check `{tempdir}/lightning_port_manager_state.json` for current allocations and use `LIGHTNING_PORT_LOCK_DIR` for isolated debugging.

**Q: Does this work with Docker/containers?**
A: Yes, as long as containers share the same filesystem (via volume mount) and use the same `LIGHTNING_PORT_LOCK_DIR`.

**Q: Why don't `_read_state()` and `_write_state()` acquire locks themselves?**
A: This is a deliberate design choice for consistency and correctness:

- **Atomicity:** The lock must be held across the entire read-modify-write cycle to prevent race conditions
- **Symmetry:** Both low-level primitives follow the same pattern (no locking), making the code easier to understand
- **Clarity:** High-level operations (public API) manage locking, low-level primitives (private) assume lock is held
- **Flexibility:** Allows high-level operations to hold lock across multiple read/write operations efficiently

If each primitive acquired its own lock, there would be a race condition between reading state and writing it back, allowing two processes to allocate the same port.

______________________________________________________________________

**Document Version:** 1.0
**Last Updated:** October 2024
**Maintainer:** LittlebullGit

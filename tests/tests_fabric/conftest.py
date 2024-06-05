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
import os
import sys
import threading
from pathlib import Path
from typing import List
from unittest.mock import Mock

import lightning.fabric
import pytest
import torch.distributed
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.strategies.launchers.subprocess_script import _ChildProcessObserver
from lightning.fabric.utilities.distributed import _destroy_dist_connection

if sys.version_info >= (3, 9):
    from concurrent.futures.process import _ExecutorManagerThread


@pytest.fixture(autouse=True)
def preserve_global_rank_variable():
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    from lightning.fabric.utilities.rank_zero import rank_zero_only as rank_zero_only_fabric
    from lightning_utilities.core.rank_zero import rank_zero_only as rank_zero_only_utilities

    functions = (rank_zero_only_fabric, rank_zero_only_utilities)
    ranks = [getattr(fn, "rank", None) for fn in functions]
    yield
    for fn, rank in zip(functions, ranks):
        if rank is not None:
            setattr(fn, "rank", rank)


@pytest.fixture(autouse=True)
def restore_env_variables():
    """Ensures that environment variables set during the test do not leak out."""
    env_backup = os.environ.copy()
    yield
    leaked_vars = os.environ.keys() - env_backup.keys()
    # restore environment as it was before running the test
    os.environ.clear()
    os.environ.update(env_backup)
    # these are currently known leakers - ideally these would not be allowed
    # TODO(fabric): this list can be trimmed, maybe PL's too after moving tests
    allowlist = {
        "CUDA_DEVICE_ORDER",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "RANK",  # set by DeepSpeed
        "CUDA_MODULE_LOADING",  # leaked by PyTorch
        "CRC32C_SW_MODE",  # set by tensorboardX
        "OMP_NUM_THREADS",  # set by our launchers
        # set by torchdynamo
        "TRITON_CACHE_DIR",
    }
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"


@pytest.fixture(autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    _destroy_dist_connection()


@pytest.fixture(autouse=True)
def thread_police_duuu_daaa_duuu_daaa():
    """Attempts to stop left-over threads to avoid test interactions."""
    active_threads_before = set(threading.enumerate())
    yield
    active_threads_after = set(threading.enumerate())

    if XLAAccelerator.is_available():
        # Ignore the check when running XLA tests for now
        return

    for thread in active_threads_after - active_threads_before:
        stop = getattr(thread, "stop", None) or getattr(thread, "exit", None)
        if thread.daemon and callable(stop):
            # A daemon thread would anyway be stopped at the end of a program
            # We do it preemptively here to reduce the risk of interactions with other tests that run after
            stop()
            assert not thread.is_alive()
        elif isinstance(thread, _ChildProcessObserver):
            thread.join(timeout=10)
        elif thread.name == "QueueFeederThread":  # tensorboardX
            thread.join(timeout=20)
        elif (
            sys.version_info >= (3, 9)
            and isinstance(thread, _ExecutorManagerThread)
            or "ThreadPoolExecutor-" in thread.name
        ):
            # probably `torch.compile`, can't narrow it down further
            continue
        else:
            raise AssertionError(f"Test left zombie thread: {thread}")


@pytest.fixture(autouse=True)
def reset_in_fabric_backward():
    """Ensures that the wrappers.in_fabric_backward global variable gets reset after each test."""
    import lightning.fabric.wrappers as wrappers

    assert hasattr(wrappers, "_in_fabric_backward")
    yield
    wrappers._in_fabric_backward = False


@pytest.fixture()
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)


@pytest.fixture()
def reset_cudnn_benchmark():
    """Ensures that the `torch.backends.cudnn.benchmark` setting gets reset before the next test runs."""
    yield
    torch.backends.cudnn.benchmark = False


def mock_xla_available(monkeypatch: pytest.MonkeyPatch, value: bool = True) -> None:
    monkeypatch.setattr(lightning.fabric.accelerators.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.plugins.environments.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.plugins.precision.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.plugins.io.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.strategies.single_xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.strategies.xla_fsdp, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.strategies.launchers.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setitem(sys.modules, "torch_xla", Mock())
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", Mock())
    monkeypatch.setitem(sys.modules, "torch_xla.experimental", Mock())
    monkeypatch.setitem(sys.modules, "torch_xla.distributed.fsdp.wrap", Mock())


@pytest.fixture()
def xla_available(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_xla_available(monkeypatch)


def mock_tpu_available(monkeypatch: pytest.MonkeyPatch, value: bool = True) -> None:
    mock_xla_available(monkeypatch, value)
    monkeypatch.setattr(lightning.fabric.accelerators.xla.XLAAccelerator, "is_available", lambda: value)
    monkeypatch.setattr(lightning.fabric.accelerators.xla.XLAAccelerator, "auto_device_count", lambda *_: 8)


@pytest.fixture()
def tpu_available(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_tpu_available(monkeypatch)


@pytest.fixture()
def caplog(caplog):
    """Workaround for https://github.com/pytest-dev/pytest/issues/3697.

    Setting ``filterwarnings`` with pytest breaks ``caplog`` when ``not logger.propagate``.

    """
    import logging

    lightning_logger = logging.getLogger("lightning.fabric")
    propagate = lightning_logger.propagate
    lightning_logger.propagate = True
    yield caplog
    lightning_logger.propagate = propagate


@pytest.fixture(autouse=True)
def leave_no_artifacts_behind():
    tests_root = Path(__file__).parent.parent
    files_before = {p for p in tests_root.rglob("*") if "__pycache__" not in p.parts}
    yield
    files_after = {p for p in tests_root.rglob("*") if "__pycache__" not in p.parts}
    difference = files_after - files_before
    difference = {str(f.relative_to(tests_root)) for f in difference}
    assert not difference, f"Test left artifacts behind: {difference}"


def pytest_collection_modifyitems(items: List[pytest.Function], config: pytest.Config) -> None:
    """An adaptation of `tests/tests_pytorch/conftest.py::pytest_collection_modifyitems`"""
    initial_size = len(items)
    conditions = []
    filtered, skipped = 0, 0

    options = {
        "standalone": "PL_RUN_STANDALONE_TESTS",
        "min_cuda_gpus": "PL_RUN_CUDA_TESTS",
        "tpu": "PL_RUN_TPU_TESTS",
    }
    if os.getenv(options["standalone"], "0") == "1" and os.getenv(options["min_cuda_gpus"], "0") == "1":
        # special case: we don't have a CPU job for standalone tests, so we shouldn't run only cuda tests.
        # by deleting the key, we avoid filtering out the CPU tests
        del options["min_cuda_gpus"]

    for kwarg, env_var in options.items():
        # this will compute the intersection of all tests selected per environment variable
        if os.getenv(env_var, "0") != "1":
            continue
        conditions.append(env_var)
        for i, test in reversed(list(enumerate(items))):  # loop in reverse, since we are going to pop items
            already_skipped = any(marker.name == "skip" for marker in test.own_markers)
            if already_skipped:
                # the test was going to be skipped anyway, filter it out
                items.pop(i)
                skipped += 1
                continue
            has_runif_with_kwarg = any(
                marker.name == "skipif" and marker.kwargs.get(kwarg) for marker in test.own_markers
            )
            if not has_runif_with_kwarg:
                # the test has `@RunIf(kwarg=True)`, filter it out
                items.pop(i)
                filtered += 1

    if config.option.verbose >= 0 and (filtered or skipped):
        writer = config.get_terminal_writer()
        writer.write(
            f"\nThe number of tests has been filtered from {initial_size} to {initial_size - filtered} after the"
            f" filters {conditions}.\n{skipped} tests are marked as unconditional skips.\nIn total, {len(items)} tests"
            " will run.\n",
            flush=True,
            bold=True,
            purple=True,  # oh yeah, branded pytest messages
        )

    # error out on our deprecation warnings - ensures the code and tests are kept up-to-date
    deprecation_error = pytest.mark.filterwarnings(
        "error::lightning.fabric.utilities.rank_zero.LightningDeprecationWarning",
    )
    for item in items:
        item.add_marker(deprecation_error)

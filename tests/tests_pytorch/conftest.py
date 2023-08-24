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
import signal
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import List
from unittest.mock import Mock

import pytest
import torch.distributed

import lightning.fabric
import lightning.pytorch
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from lightning.fabric.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_1_12
from lightning.pytorch.trainer.connectors.signal_connector import _SignalConnector
from tests_pytorch import _PATH_DATASETS


@pytest.fixture(scope="session")
def datadir():
    return Path(_PATH_DATASETS)


@pytest.fixture(autouse=True)
def preserve_global_rank_variable():
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    rank = getattr(rank_zero_only, "rank", None)
    yield
    if rank is not None:
        setattr(rank_zero_only, "rank", rank)


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
    allowlist = {
        "CUBLAS_WORKSPACE_CONFIG",  # enabled with deterministic flag
        "CUDA_DEVICE_ORDER",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "WANDB_MODE",
        "WANDB_REQUIRE_SERVICE",
        "WANDB_SERVICE",
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUStrategy
        "CUDA_MODULE_LOADING",  # leaked since PyTorch 1.13
        "KMP_INIT_AT_FORK",  # leaked since PyTorch 1.13
        "KMP_DUPLICATE_LIB_OK",  # leaked since PyTorch 1.13
        "CRC32C_SW_MODE",  # leaked by tensorboardX
        "TRITON_CACHE_DIR",  # leaked by torch.compile
        # leaked by XLA
        "ALLOW_MULTIPLE_LIBTPU_LOAD",
        "GRPC_VERBOSITY",
        "TF_CPP_MIN_LOG_LEVEL",
        "TF_GRPC_DEFAULT_OPTIONS",
        "XLA_FLAGS",
    }
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """Ensures that signal handlers get restored before the next test runs.

    This is a safety net for tests that don't run Trainer's teardown.

    """
    valid_signals = _SignalConnector._valid_signals()
    if not _IS_WINDOWS:
        # SIGKILL and SIGSTOP are not allowed to be modified by the user
        valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
    handlers = {signum: signal.getsignal(signum) for signum in valid_signals}
    yield
    for signum, handler in handlers.items():
        if handler is not None:
            signal.signal(signum, handler)


@pytest.fixture(autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture(autouse=True)
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    torch.use_deterministic_algorithms(False)


def mock_cuda_count(monkeypatch, n: int) -> None:
    monkeypatch.setattr(lightning.fabric.accelerators.cuda, "num_cuda_devices", lambda: n)
    monkeypatch.setattr(lightning.pytorch.accelerators.cuda, "num_cuda_devices", lambda: n)


@pytest.fixture()
def cuda_count_0(monkeypatch):
    mock_cuda_count(monkeypatch, 0)


@pytest.fixture()
def cuda_count_1(monkeypatch):
    mock_cuda_count(monkeypatch, 1)


@pytest.fixture()
def cuda_count_2(monkeypatch):
    mock_cuda_count(monkeypatch, 2)


@pytest.fixture()
def cuda_count_4(monkeypatch):
    mock_cuda_count(monkeypatch, 4)


def mock_mps_count(monkeypatch, n: int) -> None:
    if n > 0 and not _TORCH_GREATER_EQUAL_1_12:

        class MpsDeviceMock:
            def __new__(cls, self, *args, **kwargs):
                return "mps"

        # torch doesn't allow creation of mps devices on older versions
        monkeypatch.setattr("torch.device", MpsDeviceMock)
    monkeypatch.setattr(lightning.fabric.accelerators.mps, "_get_all_available_mps_gpus", lambda: [0] if n > 0 else [])
    monkeypatch.setattr(lightning.fabric.accelerators.mps.MPSAccelerator, "is_available", lambda *_: n > 0)


@pytest.fixture()
def mps_count_0(monkeypatch):
    mock_mps_count(monkeypatch, 0)


@pytest.fixture()
def mps_count_1(monkeypatch):
    mock_mps_count(monkeypatch, 1)


def mock_xla_available(monkeypatch: pytest.MonkeyPatch, value: bool = True) -> None:
    monkeypatch.setattr(lightning.pytorch.strategies.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.pytorch.strategies.single_xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.pytorch.plugins.precision.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.pytorch.strategies.launchers.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.accelerators.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.plugins.environments.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.plugins.io.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.strategies.xla, "_XLA_AVAILABLE", value)
    monkeypatch.setattr(lightning.fabric.strategies.launchers.xla, "_XLA_AVAILABLE", value)


@pytest.fixture()
def xla_available(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_xla_available(monkeypatch)


def mock_tpu_available(monkeypatch: pytest.MonkeyPatch, value: bool = True) -> None:
    mock_xla_available(monkeypatch, value)
    monkeypatch.setattr(lightning.pytorch.accelerators.xla.XLAAccelerator, "is_available", lambda: value)
    monkeypatch.setattr(lightning.fabric.accelerators.xla.XLAAccelerator, "is_available", lambda: value)
    monkeypatch.setattr(lightning.pytorch.accelerators.xla.XLAAccelerator, "auto_device_count", lambda *_: 8)
    monkeypatch.setattr(lightning.fabric.accelerators.xla.XLAAccelerator, "auto_device_count", lambda *_: 8)
    monkeypatch.setitem(sys.modules, "torch_xla", Mock())
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", Mock())
    monkeypatch.setitem(sys.modules, "torch_xla.experimental", Mock())


@pytest.fixture()
def tpu_available(monkeypatch) -> None:
    mock_tpu_available(monkeypatch)


@pytest.fixture()
def caplog(caplog):
    """Workaround for https://github.com/pytest-dev/pytest/issues/3697.

    Setting ``filterwarnings`` with pytest breaks ``caplog`` when ``not logger.propagate``.

    """
    import logging

    root_logger = logging.getLogger()
    root_propagate = root_logger.propagate
    root_logger.propagate = True

    propagation_dict = {
        name: logging.getLogger(name).propagate
        for name in logging.root.manager.loggerDict
        if name.startswith("lightning.pytorch")
    }
    for name in propagation_dict:
        logging.getLogger(name).propagate = True

    yield caplog

    root_logger.propagate = root_propagate
    for name, propagate in propagation_dict.items():
        logging.getLogger(name).propagate = propagate


@pytest.fixture()
def tmp_path_server(tmp_path):
    Handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    from http.server import ThreadingHTTPServer

    with ThreadingHTTPServer(("localhost", 0), Handler) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        yield server.server_address
        server.shutdown()


@pytest.fixture()
def single_process_pg():
    """Initialize the default process group with only the current process for testing purposes.

    The process group is destroyed when the with block is exited.

    """
    if torch.distributed.is_initialized():
        raise RuntimeError("Can't use `single_process_pg` when the default process group is already initialized.")

    orig_environ = os.environ.copy()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_network_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group("gloo")
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()
        os.environ.clear()
        os.environ.update(orig_environ)


def pytest_collection_modifyitems(items: List[pytest.Function], config: pytest.Config) -> None:
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
        if os.getenv(env_var, "0") == "1":
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

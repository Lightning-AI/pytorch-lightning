# Copyright The PyTorch Lightning team.
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

import pytest
import torch.distributed

from pytorch_lightning.plugins.environments.lightning_environment import find_free_network_port
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from pytorch_lightning.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_1_8
from tests import _PATH_DATASETS


@pytest.fixture(scope="session")
def datadir():
    return Path(_PATH_DATASETS)


@pytest.fixture(scope="function", autouse=True)
def preserve_global_rank_variable():
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    from pytorch_lightning.utilities.distributed import rank_zero_only

    rank = getattr(rank_zero_only, "rank", None)
    yield
    if rank is not None:
        setattr(rank_zero_only, "rank", rank)


@pytest.fixture(scope="function", autouse=True)
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
        "HOROVOD_FUSION_THRESHOLD",
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUPlugin
        # set by XLA
        "TF2_BEHAVIOR",
        "XRT_MESH_SERVICE_ADDRESS",
        "XRT_TORCH_DIST_ROOT",
        "XRT_MULTI_PROCESSING_DEVICE",
        "XRT_SHARD_WORLD_SIZE",
        "XRT_LOCAL_WORKER",
        "XRT_HOST_WORLD_SIZE",
        "XRT_SHARD_ORDINAL",
        "XRT_SHARD_LOCAL_ORDINAL",
    }
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"


@pytest.fixture(scope="function", autouse=True)
def restore_signal_handlers():
    """Ensures that signal handlers get restored before the next test runs.

    This is a safety net for tests that don't run Trainer's teardown.
    """
    valid_signals = SignalConnector._valid_signals()
    if not _IS_WINDOWS:
        # SIGKILL and SIGSTOP are not allowed to be modified by the user
        valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
    handlers = {signum: signal.getsignal(signum) for signum in valid_signals}
    yield
    for signum, handler in handlers.items():
        if handler is not None:
            signal.signal(signum, handler)


@pytest.fixture(scope="function", autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="function", autouse=True)
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    if _TORCH_GREATER_EQUAL_1_8:
        torch.use_deterministic_algorithms(False)
    else:
        torch.set_deterministic(False)


@pytest.fixture
def caplog(caplog):
    """Workaround for https://github.com/pytest-dev/pytest/issues/3697.

    Setting ``filterwarnings`` with pytest breaks ``caplog`` when ``not logger.propagate``.
    """
    import logging

    lightning_logger = logging.getLogger("pytorch_lightning")
    propagate = lightning_logger.propagate
    lightning_logger.propagate = True
    yield caplog
    lightning_logger.propagate = propagate


@pytest.fixture
def tmpdir_server(tmpdir):
    if sys.version_info >= (3, 7):
        Handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
        from http.server import ThreadingHTTPServer
    else:
        # unfortunately SimpleHTTPRequestHandler doesn't accept the directory arg in python3.6
        # so we have to hack it like this

        class Handler(SimpleHTTPRequestHandler):
            def translate_path(self, path):
                # get the path from cwd
                path = super().translate_path(path)
                # get the relative path
                relpath = os.path.relpath(path, os.getcwd())
                # return the full path from root_dir
                return os.path.join(str(tmpdir), relpath)

        # ThreadingHTTPServer was added in 3.7, so we need to define it ourselves
        from http.server import HTTPServer
        from socketserver import ThreadingMixIn

        class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

    with ThreadingHTTPServer(("localhost", 0), Handler) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        yield server.server_address
        server.shutdown()


@pytest.fixture
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


def pytest_collection_modifyitems(items):
    # filter out special tests
    if os.getenv("PL_RUN_STANDALONE_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(standalone=True)`
            if marker.name == "skipif" and marker.kwargs.get("standalone")
        ]
    elif os.getenv("PL_RUN_SLOW_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(slow=True)`
            if marker.name == "skipif" and marker.kwargs.get("slow")
        ]

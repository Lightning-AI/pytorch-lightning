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
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

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
        "WANDB_REQUIRE_SERVICE",
        "WANDB_SERVICE",
        "HOROVOD_FUSION_THRESHOLD",
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUStrategy
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
        "TF_CPP_MIN_LOG_LEVEL",
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
    Handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
    from http.server import ThreadingHTTPServer

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
    elif os.getenv("PL_RUN_IPU_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(ipu=True)`
            if marker.name == "skipif" and marker.kwargs.get("ipu")
        ]


def pytest_addoption(parser):
    parser.addoption("--hpus", action="store", type=int, default=1, help="Number of hpus 1-8")
    parser.addoption(
        "--hmp-bf16", action="store", type=str, default="./ops_bf16_mnist.txt", help="bf16 ops list file in hmp O1 mode"
    )
    parser.addoption(
        "--hmp-fp32", action="store", type=str, default="./ops_fp32_mnist.txt", help="fp32 ops list file in hmp O1 mode"
    )


@pytest.fixture
def hpus(request):
    hpus = request.config.getoption("--hpus")
    return hpus

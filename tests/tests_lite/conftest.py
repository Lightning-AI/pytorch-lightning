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
from typing import List

import pytest
import torch.distributed

import lightning_lite


@pytest.fixture(scope="function", autouse=True)
def preserve_global_rank_variable():
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    from lightning_lite.utilities.rank_zero import rank_zero_only

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
    # TODO(lite): this list can be trimmed, maybe PL's too after moving tests
    allowlist = {
        "CUDA_DEVICE_ORDER",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "HOROVOD_FUSION_THRESHOLD",
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUStrategy
    }
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"


@pytest.fixture(scope="function", autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="function")
def xla_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lightning_lite.accelerators.tpu, "_XLA_AVAILABLE", True)
    monkeypatch.setattr(lightning_lite.plugins.environments.xla_environment, "_XLA_AVAILABLE", True)
    monkeypatch.setattr(lightning_lite.strategies.xla, "_XLA_AVAILABLE", True)
    monkeypatch.setattr(lightning_lite.strategies.launchers.xla, "_XLA_AVAILABLE", True)


@pytest.fixture(scope="function")
def tpu_available(xla_available, monkeypatch) -> None:
    monkeypatch.setattr(lightning_lite.accelerators.tpu.TPUAccelerator, "is_available", lambda: True)


@pytest.fixture
def caplog(caplog):
    """Workaround for https://github.com/pytest-dev/pytest/issues/3697.

    Setting ``filterwarnings`` with pytest breaks ``caplog`` when ``not logger.propagate``.
    """
    import logging

    lightning_logger = logging.getLogger("lightning_lite")
    propagate = lightning_logger.propagate
    lightning_logger.propagate = True
    yield caplog
    lightning_logger.propagate = propagate


def pytest_collection_modifyitems(items: List[pytest.Function], config: pytest.Config) -> None:
    """An adaptation of `tests/tests_pytorch/conftest.py::pytest_collection_modifyitems`"""
    initial_size = len(items)
    conditions = []
    filtered, skipped = 0, 0

    options = dict(
        standalone="PL_RUN_STANDALONE_TESTS",
        min_cuda_gpus="PL_RUN_CUDA_TESTS",
        ipu="PL_RUN_IPU_TESTS",
        tpu="PL_RUN_TPU_TESTS",
    )
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
        "error::lightning_lite.utilities.rank_zero.LightningDeprecationWarning",
    )
    for item in items:
        item.add_marker(deprecation_error)

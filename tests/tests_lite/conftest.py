import os
from typing import List

import pytest


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

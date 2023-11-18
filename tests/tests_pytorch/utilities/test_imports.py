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

import importlib
import subprocess
import sys
from textwrap import dedent
from unittest import mock

import pytest
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from lightning_utilities.core.imports import RequirementCache
from torch.distributed import is_available

from tests_pytorch.helpers.runif import RunIf


def test_imports():
    try:
        import omegaconf  # noqa
    except ModuleNotFoundError:
        assert not _OMEGACONF_AVAILABLE
    else:
        assert _OMEGACONF_AVAILABLE


def _shortcut_patch(orig_fn, shortcut_case, attr_names=None):
    """Patch a function to return False value in the shortcut case, otherwise return original value."""

    def new_fn(*args, **kwargs):
        if attr_names is not None:
            # We assume that the first argument is the object to check and therefore values are stored in attributes.
            self = args[0]
            values = tuple(getattr(self, attr_name) for attr_name in attr_names)
        else:
            values = args
        match = True
        for value, case in zip(values, shortcut_case):
            # Go through values passed to the original function and compare them to the shortcut case.
            # We are iterating in case the `shortcut_case` and `args` lengths are different.
            if value != case:
                match = False
                break
        if match:  # If all values match the shortcut case, return False to simulate the module not being available.
            return False
        # otherwise return the original value
        return orig_fn(*args, **kwargs)

    return new_fn


@pytest.fixture()
def clean_import():
    """This fixture allows test to import {pytorch_}lightning* modules completely cleanly, regardless of the current
    state of the imported modules.

    Afterwards, it restores the original state of the modules.

    """
    import sys

    # copy modules to avoid modifying the original
    old_sys_modules = sys.modules
    # remove all *lightning* modules
    new_sys_modules = {key: value for key, value in sys.modules.items() if "lightning" not in key}
    sys.modules = new_sys_modules  # replace sys.modules with the new one
    yield
    sys.modules = old_sys_modules  # restore original modules


@pytest.mark.parametrize(
    ("patch_name", "new_fn", "to_import"),
    [
        pytest.param(
            "torch.distributed.is_available", _shortcut_patch(is_available, ()), "lightning.pytorch", id="ProcessGroup"
        ),
        pytest.param(
            "lightning_utilities.core.imports.RequirementCache.__bool__",
            _shortcut_patch(RequirementCache.__bool__, ("neptune",), ("requirement",)),
            "lightning.pytorch.loggers.neptune",
            id="neptune",
        ),
        pytest.param(
            "lightning_utilities.core.imports.RequirementCache.__bool__",
            _shortcut_patch(RequirementCache.__bool__, ("jsonargparse[signatures]>=4.12.0",), ("requirement",)),
            "lightning.pytorch.cli",
            id="cli",
        ),
    ],
)
def test_import_with_unavailable_dependencies(patch_name, new_fn, to_import, clean_import):
    """This tests simulates unavailability of certain modules by patching the functions that check for their
    availability.

    When the patch is applied and the module is imported, it should not raise any errors. The list of cases to check was
    compiled by finding else branches of top-level if statements checking for the availability of the module and
    performing imports.

    """
    with mock.patch(patch_name, new=new_fn):
        importlib.import_module(to_import)


def test_import_pytorch_lightning_with_torch_dist_unavailable():
    """Test that the package can be imported regardless of whether torch.distributed is available."""
    code = dedent(
        """
        import torch

        # pretend torch.distributed not available
        for name in list(torch.distributed.__dict__.keys()):
            if not name.startswith("__"):
                delattr(torch.distributed, name)

        torch.distributed.is_available = lambda: False

        import lightning.pytorch
        """
    )
    # run in complete isolation
    assert subprocess.call([sys.executable, "-c", code]) == 0


@RunIf(deepspeed=True)
def test_import_deepspeed_lazily():
    """Test that we are importing deepspeed only when necessary."""
    code = dedent(
        """
        import lightning.pytorch
        import sys

        assert 'deepspeed' not in sys.modules
        from lightning.pytorch.strategies import DeepSpeedStrategy
        from lightning.pytorch.plugins import DeepSpeedPrecision
        assert 'deepspeed' not in sys.modules

        import deepspeed
        assert 'deepspeed' in sys.modules
        """
    )
    # run in complete isolation
    assert subprocess.call([sys.executable, "-c", code]) == 0


@RunIf(min_python="3.9")
def test_import_lightning_multiprocessing_start_method_not_set():
    """Regression test for avoiding the lightning import to set the multiprocessing context."""
    package_name = "pytorch_lightning" if "lightning.pytorch" == "pytorch_lightning" else "lightning"

    # The following would fail with "context has already been set"
    code = dedent(
        f"""
        import sys
        import multiprocessing as mp

        import {package_name}
        mp.set_start_method("spawn")
        """
    )
    # run in complete isolation
    assert subprocess.call([sys.executable, "-c", code]) == 0

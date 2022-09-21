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

import importlib
import operator
import sys
from unittest import mock

import pytest
from lightning_utilities.core.imports import compare_version, module_available, RequirementCache
from torch.distributed import is_available

from pytorch_lightning.overrides.fairscale import _FAIRSCALE_AVAILABLE
from pytorch_lightning.strategies.bagua import _BAGUA_AVAILABLE
from pytorch_lightning.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from pytorch_lightning.utilities import _APEX_AVAILABLE, _HOROVOD_AVAILABLE, _OMEGACONF_AVAILABLE, _POPTORCH_AVAILABLE


def test_imports():
    try:
        import apex  # noqa
    except ModuleNotFoundError:
        assert not _APEX_AVAILABLE
    else:
        assert _APEX_AVAILABLE

    try:
        import bagua  # noqa
    except ModuleNotFoundError:
        assert not _BAGUA_AVAILABLE
    else:
        assert _BAGUA_AVAILABLE

    try:
        import deepspeed  # noqa
    except ModuleNotFoundError:
        assert not _DEEPSPEED_AVAILABLE
    else:
        assert _DEEPSPEED_AVAILABLE

    try:
        import fairscale.nn  # noqa
    except ModuleNotFoundError:
        assert not _FAIRSCALE_AVAILABLE
    else:
        assert _FAIRSCALE_AVAILABLE

    try:
        import horovod.torch  # noqa
    except ModuleNotFoundError:
        assert not _HOROVOD_AVAILABLE
    else:
        assert _HOROVOD_AVAILABLE

    try:
        import omegaconf  # noqa
    except ModuleNotFoundError:
        assert not _OMEGACONF_AVAILABLE
    else:
        assert _OMEGACONF_AVAILABLE

    try:
        import poptorch  # noqa
    except ModuleNotFoundError:
        assert not _POPTORCH_AVAILABLE
    else:
        assert _POPTORCH_AVAILABLE


def _shortcut_patch(orig_fn, shortcut_case, attr_names=None):
    def new_fn(*args, **kwargs):
        if attr_names is not None:
            self = args[0]
            values = tuple(getattr(self, attr_name) for attr_name in attr_names)
        else:
            values = args
        match = True
        for value, case in zip(values, shortcut_case):
            if value != case:
                match = False
                break
        if match:
            return False
        return orig_fn(*args, **kwargs)

    return new_fn


@pytest.mark.parametrize(
    ["patch_name", "new_fn", "to_import"],
    [
        ("torch.distributed.is_available", _shortcut_patch(is_available, ()), "pytorch_lightning"),
        (
            "lightning_utilities.core.imports.RequirementCache.__bool__",
            _shortcut_patch(RequirementCache.__bool__, ("neptune-client",), ("requirement",)),
            "pytorch_lightning.loggers.neptune",
        ),
        (
            "lightning_utilities.core.imports.RequirementCache.__bool__",
            _shortcut_patch(RequirementCache.__bool__, ("jsonargparse[signatures]>=4.12.0",), ("requirement",)),
            "pytorch_lightning.cli",
        ),
        (
            "lightning_utilities.core.imports.module_available",
            _shortcut_patch(module_available, ("fairscale.nn",)),
            "pytorch_lightning.strategies",
        ),
        (
            "lightning_utilities.core.imports.compare_version",
            _shortcut_patch(compare_version, ("torch", operator.ge, "1.12.0")),
            "pytorch_lightning.strategies.fully_sharded_native",
        ),
    ],
    ids=["ProcessGroup", "neptune", "cli", "fairscale", "fully_sharded_native"],
)
def test_import_with_unavailable_dependencies(patch_name, new_fn, to_import):
    pl_keys = list(
        key for key in sys.modules.keys() if key.startswith("pytorch_lightning") or key.startswith("lightning")
    )
    for pl_key in pl_keys:
        sys.modules.pop(pl_key, None)
    with mock.patch(patch_name, new=new_fn):
        importlib.import_module(to_import)

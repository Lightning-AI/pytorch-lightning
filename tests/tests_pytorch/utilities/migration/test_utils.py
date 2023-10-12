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
import glob
import logging
import os
import pickle
import sys
from unittest.mock import ANY

import lightning.pytorch as pl
import pytest
import torch
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.utilities.migration import migrate_checkpoint, pl_legacy_patch
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint, _RedirectingUnpickler
from lightning_utilities.core.imports import module_available
from lightning_utilities.test.warning import no_warning_call
from packaging.version import Version

from tests_pytorch.checkpointing.test_legacy_checkpoints import (
    CHECKPOINT_EXTENSION,
    LEGACY_BACK_COMPATIBLE_PL_VERSIONS,
    LEGACY_CHECKPOINTS_PATH,
)


def test_patch_legacy_argparse_utils():
    with pl_legacy_patch():
        from lightning.pytorch.utilities import argparse_utils

        assert callable(argparse_utils._gpus_arg_default)
        assert "lightning.pytorch.utilities.argparse_utils" in sys.modules

    assert "lightning.pytorch.utilities.argparse_utils" not in sys.modules


def test_patch_legacy_gpus_arg_default():
    with pl_legacy_patch():
        from lightning.pytorch.utilities.argparse import _gpus_arg_default

        assert callable(_gpus_arg_default)
    assert not hasattr(pl.utilities.argparse, "_gpus_arg_default")
    assert not hasattr(pl.utilities.argparse, "_gpus_arg_default")


def test_patch_legacy_fault_tolerant_mode():
    with pl_legacy_patch():
        from lightning.pytorch.utilities.enums import _FaultTolerantMode

        assert _FaultTolerantMode.AUTOMATIC.value == "automatic"
    assert not hasattr(pl.utilities.enums, "_FaultTolerantMode")


def test_test_patch_legacy_unpickler():
    with pl_legacy_patch():
        assert pickle.Unpickler == _RedirectingUnpickler
    assert pickle.Unpickler != _RedirectingUnpickler


def _list_sys_modules(pattern: str) -> str:
    return repr({k: sys.modules[k] for k in sorted(sys.modules.keys()) if pattern in k})


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@pytest.mark.skipif(module_available("lightning"), reason="This test is ONLY relevant for the STANDALONE package")
def test_test_patch_legacy_imports_standalone(pl_version):
    assert any(
        key.startswith("pytorch_lightning") for key in sys.modules
    ), f"Imported PL, so it has to be in sys.modules: {_list_sys_modules('pytorch_lightning')}"
    path_legacy = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(path_legacy, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, f'No checkpoints found in folder "{path_legacy}"'
    path_ckpt = path_ckpts[-1]

    with no_warning_call(match="Redirecting import of*"), pl_legacy_patch():
        torch.load(path_ckpt)

    assert any(
        key.startswith("pytorch_lightning") for key in sys.modules
    ), f"Imported PL, so it has to be in sys.modules: {_list_sys_modules('pytorch_lightning')}"
    assert not any(key.startswith("lightning." + "pytorch") for key in sys.modules), (
        "Did not import the unified package,"
        f" so it should not be in sys.modules: {_list_sys_modules('lightning' + '.pytorch')}"
    )


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@pytest.mark.skipif(not module_available("lightning"), reason="This test is ONLY relevant for the UNIFIED package")
def test_patch_legacy_imports_unified(pl_version):
    assert any(
        key.startswith("lightning." + "pytorch") for key in sys.modules
    ), f"Imported unified package, so it has to be in sys.modules: {_list_sys_modules('lightning' + '.pytorch')}"
    assert not any(key.startswith("pytorch_lightning") for key in sys.modules), (
        "Should not import standalone package, all imports should be redirected to the unified package;\n"
        f" environment: {_list_sys_modules('pytorch_lightning')}"
    )

    path_legacy = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(path_legacy, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, f'No checkpoints found in folder "{path_legacy}"'
    path_ckpt = path_ckpts[-1]

    # only below version 1.5.0 we pickled stuff in checkpoints
    if pl_version != "local" and Version(pl_version) < Version("1.5.0"):
        context = pytest.warns(UserWarning, match="Redirecting import of")
    else:
        context = no_warning_call(match="Redirecting import of*")
    with context, pl_legacy_patch():
        torch.load(path_ckpt)

    assert any(
        key.startswith("lightning." + "pytorch") for key in sys.modules
    ), f"Imported unified package, so it has to be in sys.modules: {_list_sys_modules('lightning' + '.pytorch')}"
    assert not any(key.startswith("pytorch_lightning") for key in sys.modules), (
        "Should not import standalone package, all imports should be redirected to the unified package;\n"
        f" environment: {_list_sys_modules('pytorch_lightning')}"
    )


def test_migrate_checkpoint(monkeypatch):
    """Test that the correct migration function gets executed given the current version of the checkpoint."""
    # A checkpoint that is older than any migration point in the index
    old_checkpoint = {"pytorch-lightning_version": "0.0.0", "content": 123}
    new_checkpoint, call_order = _run_simple_migration(monkeypatch, old_checkpoint)
    assert call_order == ["one", "two", "three", "four"]
    assert (
        new_checkpoint
        == old_checkpoint
        == {"legacy_pytorch-lightning_version": "0.0.0", "pytorch-lightning_version": pl.__version__, "content": 123}
    )

    # A checkpoint that is newer, but not the newest
    old_checkpoint = {"pytorch-lightning_version": "1.0.3", "content": 123}
    new_checkpoint, call_order = _run_simple_migration(monkeypatch, old_checkpoint)
    assert call_order == ["four"]
    assert (
        new_checkpoint
        == old_checkpoint
        == {"legacy_pytorch-lightning_version": "1.0.3", "pytorch-lightning_version": pl.__version__, "content": 123}
    )

    # A checkpoint newer than any migration point in the index
    old_checkpoint = {"pytorch-lightning_version": pl.__version__, "content": 123}
    new_checkpoint, call_order = _run_simple_migration(monkeypatch, old_checkpoint)
    assert call_order == []
    assert new_checkpoint == old_checkpoint == {"pytorch-lightning_version": pl.__version__, "content": 123}


def _run_simple_migration(monkeypatch, old_checkpoint):
    call_order = []

    def dummy_upgrade(tag):
        def upgrade(ckpt):
            call_order.append(tag)
            return ckpt

        return upgrade

    index = {
        "0.0.1": [dummy_upgrade("one")],
        "0.0.2": [dummy_upgrade("two"), dummy_upgrade("three")],
        "1.2.3": [dummy_upgrade("four")],
    }
    monkeypatch.setattr(pl.utilities.migration.utils, "_migration_index", lambda: index)
    new_checkpoint, _ = migrate_checkpoint(old_checkpoint)
    return new_checkpoint, call_order


def test_migrate_checkpoint_too_new():
    """Test checkpoint migration is a no-op with a warning when attempting to migrate a checkpoint from newer version
    of Lightning than installed."""
    super_new_checkpoint = {"pytorch-lightning_version": "99.0.0", "content": 123}
    with pytest.warns(
        PossibleUserWarning, match=f"v99.0.0, which is newer than your current Lightning version: v{pl.__version__}"
    ):
        new_checkpoint, migrations = migrate_checkpoint(super_new_checkpoint.copy())

    # no version modification
    assert not migrations
    assert new_checkpoint == super_new_checkpoint


def test_migrate_checkpoint_for_pl(caplog):
    """Test that the automatic migration in Lightning informs the user about how to make the upgrade permanent."""
    # simulate a very recent checkpoint, no migrations needed
    loaded_checkpoint = {"pytorch-lightning_version": pl.__version__, "global_step": 2, "epoch": 0}
    new_checkpoint = _pl_migrate_checkpoint(loaded_checkpoint, "path/to/ckpt")
    assert new_checkpoint == {"pytorch-lightning_version": pl.__version__, "global_step": 2, "epoch": 0}

    # simulate an old checkpoint that needed an upgrade
    loaded_checkpoint = {"pytorch-lightning_version": "0.0.1", "global_step": 2, "epoch": 0}
    with caplog.at_level(logging.INFO, logger="lightning.pytorch.utilities.migration.utils"):
        new_checkpoint = _pl_migrate_checkpoint(loaded_checkpoint, "path/to/ckpt")
    assert new_checkpoint == {
        "legacy_pytorch-lightning_version": "0.0.1",
        "pytorch-lightning_version": pl.__version__,
        "callbacks": {},
        "global_step": 2,
        "epoch": 0,
        "loops": ANY,
    }
    assert f"Lightning automatically upgraded your loaded checkpoint from v0.0.1 to v{pl.__version__}" in caplog.text


def test_migrate_checkpoint_legacy_version(monkeypatch):
    """Test that the legacy version gets set and does not change if migration is applied multiple times."""
    loaded_checkpoint = {"pytorch-lightning_version": "0.0.1", "global_step": 2, "epoch": 0}

    # pretend the current pl version is 2.0
    monkeypatch.setattr(pl, "__version__", "2.0.0")
    new_checkpoint, _ = migrate_checkpoint(loaded_checkpoint)
    assert new_checkpoint["pytorch-lightning_version"] == "2.0.0"
    assert new_checkpoint["legacy_pytorch-lightning_version"] == "0.0.1"

    # pretend the current pl version is even newer, we are migrating a second time
    monkeypatch.setattr(pl, "__version__", "3.0.0")
    new_new_checkpoint, _ = migrate_checkpoint(new_checkpoint)
    assert new_new_checkpoint["pytorch-lightning_version"] == "3.0.0"
    assert new_new_checkpoint["legacy_pytorch-lightning_version"] == "0.0.1"  # remains the same

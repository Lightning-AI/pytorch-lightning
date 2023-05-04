import glob
import os
import sys

import pytest
import torch
from lightning_utilities.core.imports import module_available
from lightning_utilities.test.warning import no_warning_call
from packaging.version import Version

from tests_pytorch.checkpointing.test_legacy_checkpoints import (
    CHECKPOINT_EXTENSION,
    LEGACY_BACK_COMPATIBLE_PL_VERSIONS,
    LEGACY_CHECKPOINTS_PATH,
)


def _list_sys_modules(pattern: str) -> str:
    return repr({k: sys.modules[k] for k in sorted(sys.modules.keys()) if pattern in k})


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@pytest.mark.skipif(
    not module_available("lightning_pytorch"), reason="This test is ONLY relevant for the STANDALONE package"
)
def test_imports_standalone(pl_version: str):
    assert any(
        key.startswith("pytorch_lightning") for key in sys.modules
    ), f"Imported PL, so it has to be in sys.modules: {_list_sys_modules('pytorch_lightning')}"
    path_legacy = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(path_legacy, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, f'No checkpoints found in folder "{path_legacy}"'
    path_ckpt = path_ckpts[-1]

    with no_warning_call(match="Redirecting import of*"):
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
def test_imports_unified(pl_version: str):
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
    with context:
        torch.load(path_ckpt)

    assert any(
        key.startswith("lightning." + "pytorch") for key in sys.modules
    ), f"Imported unified package, so it has to be in sys.modules: {_list_sys_modules('lightning' + '.pytorch')}"
    assert not any(key.startswith("pytorch_lightning") for key in sys.modules), (
        "Should not import standalone package, all imports should be redirected to the unified package;\n"
        f" environment: {_list_sys_modules('pytorch_lightning')}"
    )

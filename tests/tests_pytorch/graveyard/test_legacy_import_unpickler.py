import glob
import os
import sys

import pytest
import torch

import pytorch_lightning  # noqa: F401
from tests.tests_pytorch.helpers.utils import no_warning_call
from tests_pytorch.checkpointing.test_legacy_checkpoints import (
    CHECKPOINT_EXTENSION,
    LEGACY_BACK_COMPATIBLE_PL_VERSIONS,
    LEGACY_CHECKPOINTS_PATH,
)


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@pytest.mark.skipif(
    not "pytorch_" + "lightning" in sys.modules, reason="This test is only relevant for the standalone package"
)
def test_imports_standalone(pl_version: str):
    assert any(
        key.startswith("pytorch_" + "lightning") for key in sys.modules.keys()
    ), "Imported PL, so it has to be in sys.modules"
    path_legacy = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(path_legacy, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, f'No checkpoints found in folder "{path_legacy}"'
    path_ckpt = path_ckpts[-1]

    with no_warning_call():
        torch.load(path_ckpt)

    assert any(
        key.startswith("pytorch_" + "lightning") for key in sys.modules.keys()
    ), "Imported PL, so it has to be in sys.modules"
    assert not any(
        key.startswith("lightning.pytorch") for key in sys.modules.keys()
    ), "Did not import the unified package, so it should not be in sys.modules"


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@pytest.mark.skipif(
    "pytorch_" + "lightning" in sys.modules, reason="This test is only relevant for the unified package"
)
def test_imports_unified(pl_version: str):
    assert any(
        key.startswith("lightning.pytorch") for key in sys.modules.keys()
    ), "Imported unified package, so it has to be in sys.modules"
    assert not any(
        key.startswith("pytorch_" + "lightning") for key in sys.modules.keys()
    ), "Should not import standalone package, all imports should be redirected to the unified package"

    path_legacy = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(path_legacy, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, f'No checkpoints found in folder "{path_legacy}"'
    path_ckpt = path_ckpts[-1]

    with pytest.warns(match="Redirecting imports of"):
        torch.load(path_ckpt)

    assert any(
        key.startswith("lightning.pytorch") for key in sys.modules.keys()
    ), "Imported unified package, so it has to be in sys.modules"
    assert not any(
        key.startswith("pytorch_" + "lightning") for key in sys.modules.keys()
    ), "Should not import standalone package, all imports should be redirected to the unified package"

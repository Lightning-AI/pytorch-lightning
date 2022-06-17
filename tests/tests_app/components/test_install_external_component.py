import os
import shutil
import subprocess

import pytest

from lightning_app import _PROJECT_ROOT
from lightning_app.utilities.install_components import _pip_uninstall_component_package, install_external_component

_PACKAGE_PATH = os.path.join(_PROJECT_ROOT, "tests", "components", "sample_package_repo")
_EXTERNAL_COMPONENT_PACKAGE = "external_lightning_component_package"
_COMPONENT_PACKAGE_TAR_PATH = os.path.join(_PACKAGE_PATH, "dist", f"{_EXTERNAL_COMPONENT_PACKAGE}-0.0.1.tar.gz")


@pytest.fixture(scope="function", autouse=True)
def cleanup_installation():
    _pip_uninstall_component_package(_EXTERNAL_COMPONENT_PACKAGE.replace("_", "-"))
    shutil.rmtree(os.path.join(_PROJECT_ROOT, "lightning", "components", "myorg"), ignore_errors=True)
    yield
    _pip_uninstall_component_package(_EXTERNAL_COMPONENT_PACKAGE.replace("_", "-"))
    shutil.rmtree(os.path.join(_PACKAGE_PATH, "dist"), ignore_errors=True)
    shutil.rmtree(os.path.join(_PACKAGE_PATH, f"{_EXTERNAL_COMPONENT_PACKAGE}.egg-info"), ignore_errors=True)
    shutil.rmtree(os.path.join(_PROJECT_ROOT, "lightning", "components", "myorg"), ignore_errors=True)


@pytest.mark.usefixtures("cleanup_installation")
def test_install_external_component():
    with subprocess.Popen(
        ["python", "setup.py", "sdist"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=_PACKAGE_PATH,
    ) as proc:
        proc.wait()

    assert os.path.exists(_COMPONENT_PACKAGE_TAR_PATH)

    install_external_component(_COMPONENT_PACKAGE_TAR_PATH)

    # TODO (tchaton) Enable once stable.
    # from lightning_app.components.myorg.lightning_modules import MyCustomLightningFlow, MyCustomLightningWork

    # assert (
    #     MyCustomLightningWork.special_method()
    #     == "Hi, I'm an external lightning work component and can be added to any lightning project."
    # )
    # assert (
    #     MyCustomLightningFlow.special_method()
    #     == "Hi, I'm an external lightning flow component and can be added to any lightning project."
    # )

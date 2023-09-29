import glob
import os
from unittest import mock

import pytest
from lightning.app.testing.helpers import _RunIf
from lightning.app.utilities.git import check_github_repository, get_dir_name
from lightning.app.utilities.packaging import lightning_utils
from lightning.app.utilities.packaging.lightning_utils import (
    _prepare_lightning_wheels_and_requirements,
    _verify_lightning_version,
    get_dist_path_if_editable_install,
)
from lightning_utilities.core.imports import module_available


@pytest.mark.skipif(not module_available("lightning"), reason="TODO: should work for lightning.app too")
def test_prepare_lightning_wheels_and_requirement(tmpdir):
    """This test ensures the lightning source gets packaged inside the lightning repo."""
    package_name = "lightning"
    if not get_dist_path_if_editable_install(package_name):
        pytest.skip("Requires --editable install")

    git_dir_name = get_dir_name() if check_github_repository() else None
    if git_dir_name != package_name:
        pytest.skip("Needs to be run from within the repo")

    cleanup_handle = _prepare_lightning_wheels_and_requirements(tmpdir, package_name=package_name)
    assert len(os.listdir(tmpdir)) == 1
    assert len(glob.glob(str(tmpdir / "lightning-*.tar.gz"))) == 1

    cleanup_handle()
    assert os.listdir(tmpdir) == []


def _mocked_get_dist_path_if_editable_install(*args, **kwargs):
    return None


@mock.patch(
    "lightning.app.utilities.packaging.lightning_utils.get_dist_path_if_editable_install",
    new=_mocked_get_dist_path_if_editable_install,
)
def test_prepare_lightning_wheels_and_requirement_for_packages_installed_in_editable_mode(tmpdir):
    """This test ensures the source does not get packaged inside the lightning repo if not installed in editable
    mode."""
    cleanup_handle = _prepare_lightning_wheels_and_requirements(tmpdir)
    assert cleanup_handle is None


@pytest.mark.xfail(strict=False, reason="TODO: Find a way to check for the latest version")
@_RunIf(skip_windows=True)
def test_verify_lightning_version(monkeypatch):
    monkeypatch.setattr(lightning_utils, "__version__", "0.0.1")
    monkeypatch.setattr(lightning_utils, "_fetch_latest_version", lambda _: "0.0.2")

    # Not latest version
    with pytest.raises(Exception, match="You need to use the latest version of Lightning"):
        _verify_lightning_version()

    # Latest version
    monkeypatch.setattr(lightning_utils, "__version__", "0.0.1")
    monkeypatch.setattr(lightning_utils, "_fetch_latest_version", lambda _: "0.0.1")
    _verify_lightning_version()

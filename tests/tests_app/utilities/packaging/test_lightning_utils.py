import os
from unittest import mock

import pytest

from lightning_app.testing.helpers import RunIf
from lightning_app.utilities.packaging import lightning_utils
from lightning_app.utilities.packaging.lightning_utils import (
    _prepare_lightning_wheels_and_requirements,
    _verify_lightning_version,
)


def test_prepare_lightning_wheels_and_requirement(tmpdir):
    """This test ensures the lightning source gets packaged inside the lightning repo."""

    cleanup_handle = _prepare_lightning_wheels_and_requirements(tmpdir)
    from lightning.__version__ import version

    tar_name = f"lightning-{version}.tar.gz"
    assert sorted(os.listdir(tmpdir))[0] == tar_name
    cleanup_handle()
    assert os.listdir(tmpdir) == []


def _mocked_get_dist_path_if_editable_install(*args, **kwargs):
    return None


@mock.patch(
    "lightning_app.utilities.packaging.lightning_utils.get_dist_path_if_editable_install",
    new=_mocked_get_dist_path_if_editable_install,
)
def test_prepare_lightning_wheels_and_requirement_for_packages_installed_in_editable_mode(tmpdir):
    """This test ensures the source does not get packaged inside the lightning repo if not installed in editable
    mode."""
    cleanup_handle = _prepare_lightning_wheels_and_requirements(tmpdir)
    assert cleanup_handle is None


@pytest.mark.skip(reason="TODO: Find a way to check for the latest version")
@RunIf(skip_windows=True)
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

import os
from unittest import mock

from lightning.app.utilities.cloud import is_running_in_cloud


def test_is_running_cloud():
    """We can determine if Lightning is running in the cloud."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not is_running_in_cloud()

    with mock.patch.dict(os.environ, {"LAI_RUNNING_IN_CLOUD": "0"}, clear=True):
        assert not is_running_in_cloud()

    # in the cloud, LIGHTNING_APP_STATE_URL is defined
    with mock.patch.dict(os.environ, {"LIGHTNING_APP_STATE_URL": "defined"}, clear=True):
        assert is_running_in_cloud()

    # LAI_RUNNING_IN_CLOUD is used to fake the value of `is_running_in_cloud` when loading the app for --cloud
    with mock.patch.dict(os.environ, {"LAI_RUNNING_IN_CLOUD": "1"}):
        assert is_running_in_cloud()

import os
from unittest import mock

from lightning_app.utilities.cloud import is_running_in_cloud


@mock.patch.dict(os.environ, clear=True)
def test_is_running_locally():
    """We can determine if Lightning is running locally."""
    assert not is_running_in_cloud()


@mock.patch.dict(os.environ, {"LIGHTNING_APP_STATE_URL": "127.0.0.1"})
def test_is_running_cloud():
    """We can determine if Lightning is running in the cloud."""
    assert is_running_in_cloud()

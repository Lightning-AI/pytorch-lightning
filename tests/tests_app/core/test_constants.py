import os
from unittest import mock

from lightning.app.core.constants import get_lightning_cloud_url


@mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_URL": "https://beta.lightning.ai"})
def test_defaults():
    assert get_lightning_cloud_url() == "https://beta.lightning.ai"

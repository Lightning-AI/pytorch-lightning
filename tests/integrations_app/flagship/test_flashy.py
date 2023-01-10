import os

import pytest
from integrations_app.flagship import _PATH_INTEGRATIONS_DIR

from lightning_app.testing.testing import run_app_in_cloud


@pytest.mark.cloud
def test_app_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PATH_INTEGRATIONS_DIR, "flashy")) as (_, _, fetch_logs, _):

        print(fetch_logs)  # TODO

import os
from time import sleep

import pytest
from lightning.app.testing.testing import run_app_in_cloud

from integrations_app.public import _PATH_EXAMPLES


@pytest.mark.cloud()
def test_drive_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PATH_EXAMPLES, "drive")) as (
        _,
        _,
        fetch_logs,
        _,
    ):
        has_logs = False
        while not has_logs:
            for log in fetch_logs(["flow"]):
                if "Application End!" in log:
                    has_logs = True
            sleep(1)

import os
from subprocess import Popen
from time import sleep
from unittest import mock

import pytest
from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import run_app_in_cloud


@mock.patch.dict(os.environ, {"SKIP_LIGHTING_UTILITY_WHEELS_BUILD": "0"})
@pytest.mark.cloud
def test_commands_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PROJECT_ROOT, "examples/app_commands")) as (
        admin_page,
        _,
        fetch_logs,
        _,
    ):
        app_id = admin_page.url.split("/")[-1]
        cmd = f"lightning trigger_with_client_command --name=something --app_id {app_id}"
        Popen(cmd, shell=True).wait()
        cmd = f"lightning trigger_without_client_command --name=else --app_id {app_id}"
        Popen(cmd, shell=True).wait()

        has_logs = False
        while not has_logs:
            for log in fetch_logs(["flow"]):
                if "['something', 'else']" in log:
                    has_logs = True
            sleep(1)

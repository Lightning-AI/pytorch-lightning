import os
from subprocess import Popen
from time import sleep

import pytest
import requests
from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import run_app_in_cloud


@pytest.mark.cloud
def test_commands_and_api_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PROJECT_ROOT, "examples/app_commands_and_api")) as (
        admin_page,
        view_page,
        fetch_logs,
        _,
    ):
        app_id = admin_page.url.split("/")[-1]
        cmd = f"lightning command_with_client --name=this --app_id {app_id}"
        Popen(cmd, shell=True).wait()
        cmd = f"lightning command_without_client --name=is --app_id {app_id}"
        Popen(cmd, shell=True).wait()

        sleep(10)

        resp = requests.post(
            view_page.url.replace("/view", "") + "/user/command_without_client", json={"name": "awesome"}
        )
        assert resp.status_code == 200, resp.json()

        has_logs = False
        while not has_logs:
            for log in fetch_logs():
                if "['this', 'is', 'awesome']" in log:
                    has_logs = True
            sleep(1)

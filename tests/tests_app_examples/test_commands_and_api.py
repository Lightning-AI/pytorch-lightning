import os
from subprocess import Popen
from time import sleep

import pytest
import requests
from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import run_app_in_cloud
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.network import LightningClient


@pytest.mark.timeout(300)
@pytest.mark.cloud
def test_commands_and_api_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PROJECT_ROOT, "examples/app_commands_and_api")) as (
        admin_page,
        view_page,
        fetch_logs,
        _,
    ):
        # 1: Collect the app_id
        app_id = admin_page.url.split("/")[-1]

        # 2: Wait for App API to be ready
        client = LightningClient()
        project = _get_project(client)
        list_lightningapps = client.lightningapp_instance_service_list_lightningapp_instances(
            project_id=project.project_id
        )

        app_url = next(filter(lambda app: app.id == app_id, list_lightningapps.lightningapps)).status.url

        while True:
            sleep(10)
            resp = requests.get(app_url + "/openapi.json")
            if resp.status_code == 200:
                break

        # 3: Connect to the App
        Popen(f"python -m lightning connect {app_id} -y", shell=True).wait()

        # 4: Send the first command with the client
        cmd = "python -m lightning command with client --name=this"
        Popen(cmd, shell=True).wait()

        # 5: Send the second command without a client
        cmd = "python -m lightning command without client --name=is"
        Popen(cmd, shell=True).wait()

        # This prevents some flakyness in the CI. Couldn't reproduce it locally.
        sleep(5)

        # 6: Send a request to the Rest API directly.
        base_url = view_page.url.replace("/view", "").replace("/child_flow", "")
        resp = requests.post(base_url + "/user/command_without_client?name=awesome")
        assert resp.status_code == 200, resp.json()

        # 7: Validate the logs.
        has_logs = False
        while not has_logs:
            for log in fetch_logs():
                if "['this', 'is', 'awesome']" in log:
                    has_logs = True
            sleep(1)

        # 8: Disconnect from the App
        Popen("lightning disconnect", shell=True).wait()

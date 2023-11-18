import os
from subprocess import Popen
from time import sleep

import pytest
from lightning.app.testing.testing import run_app_in_cloud

from integrations_app.public import _PATH_EXAMPLES


@pytest.mark.timeout(300)
@pytest.mark.cloud()
def test_commands_and_api_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PATH_EXAMPLES, "commands_and_api")) as (
        _,
        view_page,
        fetch_logs,
        app_name,
    ):
        # Connect to the App and send the first & second command with the client
        # Requires to be run within the same process.
        cmd_1 = f"python -m lightning connect app {app_name}"
        cmd_2 = "python -m lightning command with client --name=this"
        cmd_3 = "python -m lightning command without client --name=is"
        cmd_4 = "python -m lightning command without client --name=awesome"
        cmd_5 = "lightning logout"
        process = Popen(" && ".join([cmd_1, cmd_2, cmd_3, cmd_4, cmd_5]), shell=True)
        process.wait()
        "/".join(view_page.url.split("/")[:-2])

        # Validate the logs.
        has_logs = False
        while not has_logs:
            for log in fetch_logs():
                if "['this', 'is', 'awesome']" in log:
                    has_logs = True
            sleep(1)

import os
from time import sleep

import pytest
from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import run_app_in_cloud


@pytest.mark.cloud
def test_custom_work_dependencies_example_cloud() -> None:
    # if requirements not installed, the app will fail
    with run_app_in_cloud(
        os.path.join(_PROJECT_ROOT, "tests/tests_app_examples/custom_work_dependencies/"),
        app_name="app.py",
    ) as (_, _, fetch_logs, _):
        has_logs = False
        while not has_logs:
            for log in fetch_logs(["flow"]):
                if "Custom Work Dependency checker End" in log:
                    has_logs = True
            sleep(1)

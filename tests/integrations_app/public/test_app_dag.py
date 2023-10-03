import os
from time import sleep

import pytest
from lightning.app.testing.testing import run_app_in_cloud

from integrations_app.public import _PATH_EXAMPLES


@pytest.mark.cloud()
def test_app_dag_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PATH_EXAMPLES, "dag")) as (_, _, fetch_logs, _):
        launch_log, finish_log = False, False
        while not (launch_log and finish_log):
            for log in fetch_logs(["flow"]):
                if "Launching a new DAG" in log:
                    launch_log = True
                elif "Finished training and evaluating" in log:
                    finish_log = True
            sleep(1)

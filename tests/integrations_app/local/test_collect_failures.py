import os
from time import sleep

import pytest
from lightning.app.testing.testing import run_app_in_cloud

from integrations_app.local import _PATH_APPS


@pytest.mark.cloud()
def test_collect_failures_example_cloud() -> None:
    # logs are in order
    expected_logs = [
        "useless_garbage_log_that_is_always_there_to_overload_logs",
        "waiting_for_work_to_be_ready",
        "work_is_running",
        "flow_and_work_are_running",
        "logger_flow_work",
        "good_value_of_i_1",
        "good_value_of_i_2",
        "good_value_of_i_3",
        "good_value_of_i_4",
        "invalid_value_of_i_5",
    ]
    with run_app_in_cloud(os.path.join(_PATH_APPS, "collect_failures")) as (
        _,
        _,
        fetch_logs,
        _,
    ):
        last_found_log_index = -1
        while len(expected_logs) != 0:
            for index, log in enumerate(fetch_logs()):
                if expected_logs[0] in log:
                    print(f"found expected log: {expected_logs[0]}")
                    expected_logs.pop(0)
                    assert index > last_found_log_index
                    if len(expected_logs) == 0:
                        break
            sleep(1)

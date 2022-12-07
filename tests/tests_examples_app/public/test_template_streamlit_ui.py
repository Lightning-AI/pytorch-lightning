import os
from time import sleep

import pytest
from tests_examples_app.public import _PATH_EXAMPLES

from lightning_app.testing.testing import run_app_in_cloud


@pytest.mark.cloud
def test_template_streamlit_ui_example_cloud() -> None:
    """This test ensures streamlit works in the cloud by clicking a button and checking the logs."""
    with run_app_in_cloud(os.path.join(_PATH_EXAMPLES, "app_template_streamlit_ui")) as (
        _,
        view_page,
        fetch_logs,
        _,
    ):
        button = view_page.frame_locator("iframe").locator('button:has-text("Should print to the terminal ?")')
        button.click()

        has_logs = False
        while not has_logs:
            for log in fetch_logs():
                print(log)
                if "Hello World!" in log:
                    has_logs = True
            sleep(1)

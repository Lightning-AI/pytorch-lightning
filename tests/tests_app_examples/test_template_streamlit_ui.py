import os
from time import sleep

import pytest
from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import run_app_in_cloud, wait_for


@pytest.mark.cloud
def test_template_streamlit_ui_example_cloud() -> None:
    """This test ensures streamlit works in the cloud by clicking a button and checking the logs."""
    with run_app_in_cloud(os.path.join(_PROJECT_ROOT, "examples/app_template_streamlit_ui")) as (
        _,
        view_page,
        fetch_logs,
        _,
    ):

        def click_button(*_, **__):
            button = view_page.frame_locator("iframe").locator('button:has-text("Should print to the terminal ?")')
            button.wait_for(timeout=5 * 1000)
            if button.all_text_contents() == ["Should print to the terminal ?"]:
                button.click()
                return True

        wait_for(view_page, click_button)

        has_logs = False
        while not has_logs:
            for log in fetch_logs():
                if "0: Hello World!" in log:
                    has_logs = True
            sleep(1)

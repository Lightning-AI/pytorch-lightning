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
        import playwright

        print("Reached")

        i = 0

        while i < 2:
            try:
                print("clicking")
                button = view_page.frame_locator("iframe").locator('button:has-text("Should print to the terminal ?")')
                button.click()
                print("clicked")
                break
            except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError) as e:
                print(e)
                try:
                    sleep(5)
                    view_page.reload()
                except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError) as e:
                    print(e)
                    pass
                sleep(2)
                i = i + 1

        has_logs = False
        while not has_logs:
            for log in fetch_logs():
                print(log)
                if "Hello World!" in log:
                    has_logs = True
            sleep(1)

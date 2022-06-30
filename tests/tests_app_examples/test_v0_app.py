import os
from time import sleep
from typing import Tuple

import pytest
from tests_app import _PROJECT_ROOT

from lightning_app.testing.testing import application_testing, LightningTestApp, run_app_in_cloud, wait_for
from lightning_app.utilities.enum import AppStage


class LightningAppTestInt(LightningTestApp):
    def run_once(self) -> Tuple[bool, float]:
        if self.root.counter > 1:
            print("V0 App End")
            self.stage = AppStage.STOPPING
            return True, 0.0
        return super().run_once()


def test_v0_app_example():
    command_line = [
        os.path.join(_PROJECT_ROOT, "examples/app_v0/app.py"),
        "--blocking",
        "False",
        "--open-ui",
        "False",
    ]
    result = application_testing(LightningAppTestInt, command_line)
    assert result.exit_code == 0


@pytest.mark.cloud
def test_v0_app_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PROJECT_ROOT, "examples/app_v0")) as (
        _,
        view_page,
        fetch_logs,
    ):

        def check_content(button_name, text_content):
            button = view_page.locator(f'button:has-text("{button_name}")')
            button.wait_for(timeout=3 * 1000)
            button.click()
            view_page.reload()
            locator = view_page.frame_locator("iframe").locator("div")
            locator.wait_for(timeout=3 * 1000)
            assert text_content in " ".join(locator.all_text_contents())
            return True

        wait_for(view_page, check_content, "TAB_1", "Hello from component A")
        wait_for(view_page, check_content, "TAB_2", "Hello from component B")

        has_logs = False
        while not has_logs:
            for log in fetch_logs():
                if "'a': 'a', 'b': 'b'" in log:
                    has_logs = True
            sleep(1)

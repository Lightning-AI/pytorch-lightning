import os
from time import sleep
from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest
from lightning.app import LightningApp
from lightning.app.runners import CloudRuntime
from lightning.app.testing import EmptyFlow
from lightning.app.testing.testing import LightningTestApp, application_testing, run_app_in_cloud, wait_for
from lightning.app.utilities.enum import AppStage
from lightning.app.utilities.load_app import load_app_from_file

from integrations_app.public import _PATH_EXAMPLES


class LightningAppTestInt(LightningTestApp):
    def run_once(self) -> Tuple[bool, float]:
        if self.root.counter == 1:
            print("V0 App End")
            self.stage = AppStage.STOPPING
            return True, 0.0
        return super().run_once()


def test_v0_app_example():
    command_line = [
        os.path.join(_PATH_EXAMPLES, "v0", "app.py"),
        "--blocking",
        "False",
        "--open-ui",
        "False",
    ]
    result = application_testing(LightningAppTestInt, command_line)
    assert result.exit_code == 0


def run_v0_app(fetch_logs, view_page):
    def check_content(button_name, text_content):
        button = view_page.locator(f'button:has-text("{button_name}")')
        button.wait_for(timeout=3 * 1000)
        button.click()
        view_page.reload()
        locator = view_page.frame_locator("iframe").locator("div")
        locator.wait_for(timeout=3 * 1000)
        assert text_content in " ".join(locator.all_text_contents())
        print(f"Validated {button_name}")
        return True

    wait_for(view_page, check_content, "TAB_1", "Hello from component A")
    wait_for(view_page, check_content, "TAB_2", "Hello from component B")
    has_logs = False
    while not has_logs:
        for log in fetch_logs(["flow"]):
            print(log)
            if "'a': 'a', 'b': 'b'" in log:
                has_logs = True
        sleep(1)


@pytest.mark.cloud()
@pytest.mark.skipif(
    os.environ.get("LIGHTNING_BYOC_CLUSTER_ID") is None,
    reason="missing LIGHTNING_BYOC_CLUSTER_ID environment variable",
)
def test_v0_app_example_byoc_cloud() -> None:
    with run_app_in_cloud(
        os.path.join(_PATH_EXAMPLES, "v0"),
        extra_args=["--cluster-id", os.environ.get("LIGHTNING_BYOC_CLUSTER_ID")],
    ) as (_, view_page, fetch_logs, _):
        run_v0_app(fetch_logs, view_page)


@pytest.mark.cloud()
def test_v0_app_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PATH_EXAMPLES, "v0")) as (
        _,
        view_page,
        fetch_logs,
        _,
    ):
        run_v0_app(fetch_logs, view_page)


@mock.patch(
    "lightning.app.runners.cloud.load_app_from_file",
    MagicMock(side_effect=ModuleNotFoundError("Module X not found")),
)
def test_load_app_from_file_module_error():
    empty_app = CloudRuntime.load_app_from_file(os.path.join(_PATH_EXAMPLES, "v0", "app.py"))
    assert isinstance(empty_app, LightningApp)
    assert isinstance(empty_app.root, EmptyFlow)


def test_load_app_from_file():
    app = load_app_from_file(os.path.join(_PATH_EXAMPLES, "v0", "app.py"))
    assert isinstance(app, LightningApp)

import logging
import os
from unittest import mock

import pytest
from click.testing import CliRunner
from lightning.app import LightningApp
from lightning.app.cli.lightning_cli import run_app
from lightning.app.testing.helpers import _RunIf
from lightning.app.testing.testing import run_app_in_cloud, wait_for

from integrations_app.public import _PATH_EXAMPLES


class QuickStartApp(LightningApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root.serve_work._parallel = True

    def run_once(self):
        done = super().run_once()
        if self.root.train_work.best_model_path:
            return True
        return done


# TODO: Investigate why it doesn't work
@pytest.mark.xfail(strict=False, reason="test is skipped because CI was blocking all the PRs.")
@_RunIf(pl=True, skip_windows=True, skip_linux=True)
def test_quick_start_example(caplog, monkeypatch):
    """This test ensures the Quick Start example properly train and serve PyTorch Lightning."""
    monkeypatch.setattr("logging.getLogger", mock.MagicMock(return_value=logging.getLogger()))

    with caplog.at_level(logging.INFO):
        with mock.patch("lightning.app.LightningApp", QuickStartApp):
            runner = CliRunner()
            result = runner.invoke(
                run_app,
                [
                    os.path.join(_PATH_EXAMPLES, "lightning-quick-start", "app.py"),
                    "--blocking",
                    "False",
                    "--open-ui",
                    "False",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0


@pytest.mark.cloud()
def test_quick_start_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PATH_EXAMPLES, "lightning-quick-start")) as (_, view_page, _, _):

        def click_gradio_demo(*_, **__):
            button = view_page.locator('button:has-text("Interactive demo")')
            button.wait_for(timeout=5 * 1000)
            button.click()
            return True

        wait_for(view_page, click_gradio_demo)

        def check_examples(*_, **__):
            locator = view_page.frame_locator("iframe").locator('button:has-text("Submit")')
            locator.wait_for(timeout=10 * 1000)
            if len(locator.all_text_contents()) > 0:
                return True
            return None

        wait_for(view_page, check_examples)

import os

import pytest
from click.testing import CliRunner
from tests_app import _PROJECT_ROOT

from lightning_app.cli.lightning_cli import logs
from lightning_app.testing.testing import run_app_in_cloud, wait_for


@pytest.mark.cloud
def test_boring_app_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PROJECT_ROOT, "examples/app_boring/"), app_name="app_dynamic.py") as (
        _,
        view_page,
        _,
        name,
    ):

        def check_hello_there(*_, **__):
            locator = view_page.frame_locator("iframe").locator('ul:has-text("Hello there!")')
            locator.wait_for(timeout=3 * 1000)
            if len(locator.all_text_contents()):
                return True

        wait_for(view_page, check_hello_there)

        runner = CliRunner()
        result = runner.invoke(logs, [name])
        lines = result.output.splitlines()

        assert result.exit_code == 0
        assert result.exception is None
        assert len(lines) > 1, result.output
        # We know that at some point we need to intstall lightning, so we check for that
        assert any(
            "Successfully built lightning" in line for line in lines
        ), f"Did not find logs with lightning installation: {result.output}"

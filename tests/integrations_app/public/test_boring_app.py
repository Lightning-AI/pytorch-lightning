import os

import pytest
from click.testing import CliRunner

from integrations_app.public import _PATH_EXAMPLES
from lightning.app.cli.lightning_cli import show
from lightning.app.testing.testing import run_app_in_cloud, wait_for


@pytest.mark.cloud
def test_boring_app_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PATH_EXAMPLES, "app_boring"), app_name="app_dynamic.py", debug=True) as (
        _,
        view_page,
        _,
        name,
    ):

        def check_hello_there(*_, **__):
            locator = view_page.frame_locator("iframe").locator('ul:has-text("Hello there!")')
            if len(locator.all_text_contents()):
                return True

        wait_for(view_page, check_hello_there)

        runner = CliRunner()
        result = runner.invoke(show.commands["logs"], [name])
        lines = result.output.splitlines()

        assert result.exit_code == 0
        assert result.exception is None
        assert any("Received from root.dict.dst_w" in line for line in lines)
        print("Succeeded App!")

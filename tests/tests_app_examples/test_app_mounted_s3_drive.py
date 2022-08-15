import os

import pytest
from click.testing import CliRunner
from tests_app import _PROJECT_ROOT

from lightning_app.cli.lightning_cli import logs
from lightning_app.testing.testing import run_app_in_cloud


@pytest.mark.cloud
def test_boring_app_example_cloud() -> None:
    with run_app_in_cloud(os.path.join(_PROJECT_ROOT, "examples/app_mounted_s3_drive/"), app_name="app.py") as (
        _,
        _,
        fetch_logs,
        name,
    ):
        for _ in fetch_logs():
            pass

        runner = CliRunner()
        result = runner.invoke(logs, [name])
        lines = result.output.splitlines()

        assert result.exit_code == 0
        assert result.exception is None
        assert any("verifications complete! Exiting work run..." in line for line in lines)
        assert any("Drive2(id=ryft-public-sample-data/) was mounted successfully!")

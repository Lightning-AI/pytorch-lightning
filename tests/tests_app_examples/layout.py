import os

from click.testing import CliRunner
from tests_app import _PROJECT_ROOT

from lightning_app.cli.lightning_cli import run_app


def test_layout_example():
    runner = CliRunner()
    result = runner.invoke(
        run_app,
        [
            os.path.join(_PROJECT_ROOT, "examples/app_layout/app.py"),
            "--blocking",
            "False",
            "--open-ui",
            "False",
        ],
        catch_exceptions=False,
    )
    assert "Layout End" in str(result.stdout_bytes)
    assert result.exit_code == 0

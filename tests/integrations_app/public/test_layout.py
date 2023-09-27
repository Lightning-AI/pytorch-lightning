import os

from click.testing import CliRunner
from lightning.cli.lightning_cli import run_app

from integrations_app.public import _PATH_EXAMPLES


def test_layout_example():
    runner = CliRunner()
    result = runner.invoke(
        run_app,
        [
            os.path.join(_PATH_EXAMPLES, "layout", "app.py"),
            "--blocking",
            "False",
            "--open-ui",
            "False",
        ],
        catch_exceptions=False,
    )
    assert "Layout End" in str(result.stdout_bytes)
    assert result.exit_code == 0

import os

from click.testing import CliRunner
from tests_app import _PROJECT_ROOT

from lightning_app.cli.lightning_cli import run_app


def test_core_features_app_example():
    runner = CliRunner()
    result = runner.invoke(
        run_app,
        [
            os.path.join(_PROJECT_ROOT, "tests/tests_app_examples/core_features_app/app.py"),
            "--blocking",
            "False",
            "--open-ui",
            "False",
            "--env",  # this is to test env variable
            "FOO=bar",
            "--env",
            "BLA=bloz",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

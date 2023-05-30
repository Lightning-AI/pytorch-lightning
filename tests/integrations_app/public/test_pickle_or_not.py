import os

import pytest
from click.testing import CliRunner

from integrations_app.public import _PATH_EXAMPLES
from lightning.app.cli.lightning_cli import run_app


# TODO: Investigate why it doesn't work
@pytest.mark.skip(reason="test has been ignored for a while and seems not to be working :(")
def test_pickle_or_not_example():
    runner = CliRunner()
    result = runner.invoke(
        run_app,
        [
            os.path.join(_PATH_EXAMPLES, "pickle_or_not", "app.py"),
            "--blocking",
            "False",
            "--open-ui",
            "False",
        ],
        catch_exceptions=False,
    )
    assert "Pickle or Not End" in str(result.stdout_bytes)
    assert result.exit_code == 0

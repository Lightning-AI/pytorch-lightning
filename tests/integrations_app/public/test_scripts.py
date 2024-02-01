import os

import pytest
from click.testing import CliRunner
from lightning.app.cli.lightning_cli import run_app
from lightning.app.testing.helpers import _run_script, _RunIf

from integrations_app.public import _PATH_EXAMPLES


@_RunIf(pl=True, skip_windows=True)
@pytest.mark.parametrize(
    "file",
    [
        pytest.param("component_tracer.py"),
        pytest.param("component_popen.py"),
    ],
)
def test_scripts(file):
    _run_script(str(os.path.join(_PATH_EXAMPLES, f"components/python/{file}")))


@pytest.mark.xfail(strict=False, reason="causing some issues with CI, not sure if the test is actually needed")
@_RunIf(pl=True, skip_windows=True)
def test_components_app_example():
    runner = CliRunner()
    result = runner.invoke(
        run_app,
        [
            os.path.join(_PATH_EXAMPLES, "components/python/app.py"),
            "--blocking",
            "False",
            "--open-ui",
            "False",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "tracer script succeed" in result.stdout

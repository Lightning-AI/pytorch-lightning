import logging
import os
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from lightning_app import _PROJECT_ROOT, LightningApp
from lightning_app.cli.lightning_cli import _run_app, run_app
from lightning_app.runners.runtime_type import RuntimeType
from lightning_app.testing.helpers import RunIf
from lightning_app.utilities.app_helpers import convert_print_to_logger_info


@RunIf(skip_linux=True)
@mock.patch("click.launch")
@pytest.mark.parametrize("open_ui", (True, False))
def test_lightning_run_app(lauch_mock: mock.MagicMock, open_ui, caplog, monkeypatch):
    """This test validates the command is runned properly and the LightningApp method is being executed."""

    monkeypatch.setattr("lightning_app._logger", logging.getLogger())

    original_method = LightningApp._run

    @convert_print_to_logger_info
    def _lightning_app_run_and_logging(self, *args, **kwargs):
        original_method(self, *args, **kwargs)
        print("1" if open_ui else "0")
        print(self)

    with caplog.at_level(logging.INFO):
        with mock.patch("lightning_app.LightningApp._run", _lightning_app_run_and_logging):

            runner = CliRunner()
            result = runner.invoke(
                run_app,
                [
                    os.path.join(_PROJECT_ROOT, "tests/core/scripts/app_metadata.py"),
                    "--blocking",
                    "False",
                    "--open-ui",
                    str(open_ui),
                ],
                catch_exceptions=False,
            )
            # capture logs.
            if open_ui:
                lauch_mock.assert_called_with("http://127.0.0.1:7501/view")
            else:
                lauch_mock.assert_not_called()
        assert result.exit_code == 0
    assert len(caplog.messages) == 2
    assert bool(int(caplog.messages[0])) is open_ui


@mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_URL": "https://beta.lightning.ai"})
@mock.patch("lightning_app.cli.lightning_cli.dispatch")
@pytest.mark.parametrize("open_ui", (True, False))
def test_lightning_run_app_cloud(mock_dispatch: mock.MagicMock, open_ui, caplog, monkeypatch):
    """This test validates the command has ran properly when --cloud argument is passed.

    It tests it by checking if the click.launch is called with the right url if --open-ui was true and also checks the
    call to `dispatch` for the right arguments
    """
    monkeypatch.setattr("lightning_app.runners.cloud.logger", logging.getLogger())

    with caplog.at_level(logging.INFO):
        _run_app(
            file=os.path.join(_PROJECT_ROOT, "tests/core/scripts/app_metadata.py"),
            cloud=True,
            without_server=False,
            name="",
            blocking=False,
            open_ui=open_ui,
            no_cache=True,
            env=("FOO=bar",),
        )
    # capture logs.
    # TODO(yurij): refactor the test, check if the actual HTTP request is being sent and that the proper admin
    #  page is being opened
    mock_dispatch.assert_called_with(
        Path(os.path.join(_PROJECT_ROOT, "tests/core/scripts/app_metadata.py")),
        RuntimeType.CLOUD,
        start_server=True,
        blocking=False,
        on_before_run=mock.ANY,
        name="",
        no_cache=True,
        env_vars={"FOO": "bar"},
    )

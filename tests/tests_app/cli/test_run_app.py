import logging
import os
from pathlib import Path
from unittest import mock

import click
import lightning.app.core.constants as constants
import pytest
from click.testing import CliRunner
from lightning.app import LightningApp
from lightning.app.cli.lightning_cli import _run_app, run_app
from lightning.app.runners.runtime_type import RuntimeType
from lightning.app.utilities.app_helpers import convert_print_to_logger_info

from tests_app import _PROJECT_ROOT


@mock.patch("click.launch")
@pytest.mark.parametrize("open_ui", [True, False])
def test_lightning_run_app(lauch_mock: mock.MagicMock, open_ui, caplog, monkeypatch):
    """This test validates the command is runned properly and the LightningApp method is being executed."""
    monkeypatch.setattr("lightning.app._logger", logging.getLogger())

    original_method = LightningApp._run

    @convert_print_to_logger_info
    def _lightning_app_run_and_logging(self, *args, **kwargs):
        original_method(self, *args, **kwargs)
        print("1" if open_ui else "0")
        print(self)

    with caplog.at_level(logging.INFO):
        with mock.patch("lightning.app.LightningApp._run", _lightning_app_run_and_logging):
            runner = CliRunner()
            pytest_env = os.environ.pop("PYTEST_CURRENT_TEST")
            try:
                result = runner.invoke(
                    run_app,
                    [
                        os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py"),
                        "--blocking",
                        "False",
                        "--open-ui",
                        str(open_ui),
                    ],
                    catch_exceptions=False,
                )
            finally:
                os.environ["PYTEST_CURRENT_TEST"] = pytest_env
            # capture logs.
            if open_ui:
                # Get the designated port
                port = constants.APP_SERVER_PORT

                lauch_mock.assert_called_with(f"http://127.0.0.1:{port}/view")
            else:
                lauch_mock.assert_not_called()
        assert result.exit_code == 0
    assert len(caplog.messages) == 4
    assert bool(int(caplog.messages[0])) is open_ui


@mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_URL": "https://beta.lightning.ai"})
@mock.patch("lightning.app.cli.lightning_cli.dispatch")
@pytest.mark.parametrize("open_ui", [True, False])
def test_lightning_run_app_cloud(mock_dispatch: mock.MagicMock, open_ui, caplog, monkeypatch):
    """This test validates the command has ran properly when --cloud argument is passed.

    It tests it by checking if the click.launch is called with the right url if --open-ui was true and also checks the
    call to `dispatch` for the right arguments.

    """
    monkeypatch.setattr("lightning.app.runners.cloud.logger", logging.getLogger())

    with caplog.at_level(logging.INFO):
        _run_app(
            file=os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py"),
            cloud=True,
            without_server=False,
            name="",
            blocking=False,
            open_ui=open_ui,
            no_cache=True,
            env=("FOO=bar",),
            secret=("BAR=my-secret",),
            run_app_comment_commands=False,
            enable_basic_auth="",
        )

        # Get the designated port
        port = constants.APP_SERVER_PORT

    # capture logs.
    # TODO(yurij): refactor the test, check if the actual HTTP request is being sent and that the proper admin
    #  page is being opened
    mock_dispatch.assert_called_with(
        Path(os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py")),
        RuntimeType.CLOUD,
        start_server=True,
        blocking=False,
        open_ui=open_ui,
        name="",
        no_cache=True,
        env_vars={"FOO": "bar"},
        secrets={"BAR": "my-secret"},
        run_app_comment_commands=False,
        enable_basic_auth="",
        port=port,
    )


@mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_URL": "https://beta.lightning.ai"})
@mock.patch("lightning.app.cli.lightning_cli.dispatch")
@pytest.mark.parametrize("open_ui", [True, False])
def test_lightning_run_app_cloud_with_run_app_commands(mock_dispatch: mock.MagicMock, open_ui, caplog, monkeypatch):
    """This test validates the command has ran properly when --cloud argument is passed.

    It tests it by checking if the click.launch is called with the right url if --open-ui was true and also checks the
    call to `dispatch` for the right arguments.

    """
    monkeypatch.setattr("lightning.app.runners.cloud.logger", logging.getLogger())

    with caplog.at_level(logging.INFO):
        _run_app(
            file=os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py"),
            cloud=True,
            without_server=False,
            name="",
            blocking=False,
            open_ui=open_ui,
            no_cache=True,
            env=("FOO=bar",),
            secret=("BAR=my-secret",),
            run_app_comment_commands=True,
            enable_basic_auth="",
        )

        # Get the designated port
        port = constants.APP_SERVER_PORT

    # capture logs.
    # TODO(yurij): refactor the test, check if the actual HTTP request is being sent and that the proper admin
    #  page is being opened
    mock_dispatch.assert_called_with(
        Path(os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py")),
        RuntimeType.CLOUD,
        start_server=True,
        blocking=False,
        open_ui=open_ui,
        name="",
        no_cache=True,
        env_vars={"FOO": "bar"},
        secrets={"BAR": "my-secret"},
        run_app_comment_commands=True,
        enable_basic_auth="",
        port=port,
    )


def test_lightning_run_app_secrets(monkeypatch):
    """Validates that running apps only supports the `--secrets` argument if the `--cloud` argument is passed."""
    monkeypatch.setattr("lightning.app.runners.cloud.logger", logging.getLogger())

    with pytest.raises(click.exceptions.ClickException):
        _run_app(
            file=os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py"),
            cloud=False,
            without_server=False,
            name="",
            blocking=False,
            open_ui=False,
            no_cache=True,
            env=(),
            secret=("FOO=my-secret"),
            run_app_comment_commands=False,
            enable_basic_auth="",
        )


@mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_URL": "https://beta.lightning.ai"})
@mock.patch("lightning.app.cli.lightning_cli.dispatch")
def test_lightning_run_app_enable_basic_auth_passed(mock_dispatch: mock.MagicMock, caplog, monkeypatch):
    """This test just validates the command has ran properly when --enable-basic-auth argument is passed.

    It checks the call to `dispatch` for the right arguments.

    """
    monkeypatch.setattr("lightning.app.runners.cloud.logger", logging.getLogger())

    with caplog.at_level(logging.INFO):
        _run_app(
            file=os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py"),
            cloud=True,
            without_server=False,
            name="",
            blocking=False,
            open_ui=False,
            no_cache=True,
            env=("FOO=bar",),
            secret=("BAR=my-secret",),
            run_app_comment_commands=False,
            enable_basic_auth="username:password",
        )

        # Get the designated port
        port = constants.APP_SERVER_PORT

    mock_dispatch.assert_called_with(
        Path(os.path.join(_PROJECT_ROOT, "tests/tests_app/core/scripts/app_metadata.py")),
        RuntimeType.CLOUD,
        start_server=True,
        blocking=False,
        open_ui=False,
        name="",
        no_cache=True,
        env_vars={"FOO": "bar"},
        secrets={"BAR": "my-secret"},
        run_app_comment_commands=False,
        enable_basic_auth="username:password",
        port=port,
    )

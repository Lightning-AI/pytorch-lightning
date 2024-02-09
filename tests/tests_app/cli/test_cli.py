import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from lightning.app import __version__
from lightning.app.cli.lightning_cli import _main, login, logout, run
from lightning.app.cli.lightning_cli_delete import delete
from lightning.app.cli.lightning_cli_list import get_list, list_apps
from lightning.app.utilities.exceptions import _ApiExceptionHandler


@pytest.mark.parametrize("command", [_main, run, get_list, delete])
def test_commands(command):
    runner = CliRunner()
    result = runner.invoke(command)
    assert result.exit_code == 0


def test_main_lightning_cli_no_arguments():
    """Validate the Lightning CLI without args."""
    res = os.popen("lightning_app").read()
    assert "login   " in res
    assert "logout  " in res
    assert "run     " in res
    assert "list    " in res
    assert "delete  " in res
    assert "show    " in res


def test_main_lightning_cli_help():
    """Validate the Lightning CLI."""
    res = os.popen("lightning_app --help").read()
    assert "login   " in res
    assert "logout  " in res
    assert "run     " in res
    assert "list    " in res
    assert "delete  " in res
    assert "show    " in res

    res = os.popen("lightning_app run --help").read()
    assert "app  " in res

    # hidden run commands should not appear in the help text
    assert "server" not in res
    assert "flow" not in res
    assert "work" not in res
    assert "frontend" not in res

    # inspect show group
    res = os.popen("lightning_app show --help").read()
    assert "logs " in res


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.cli.cmd_apps._AppManager.list")
def test_list_apps(list_command: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(list_apps)


@mock.patch("lightning.app.utilities.login.Auth._run_server")
@mock.patch("lightning.app.utilities.login.Auth.clear")
def test_cli_login(clear: mock.MagicMock, run_server: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(login)

    clear.assert_called_once_with()
    run_server.assert_called_once()


@mock.patch("pathlib.Path.unlink")
@mock.patch("pathlib.Path.exists")
@pytest.mark.parametrize("creds", [True, False])
def test_cli_logout(exists: mock.MagicMock, unlink: mock.MagicMock, creds: bool):
    exists.return_value = creds
    runner = CliRunner()
    runner.invoke(logout)

    exists.assert_called_once_with()
    if creds:
        unlink.assert_called_once_with()
    else:
        unlink.assert_not_called()


def test_lightning_cli_version():
    res = os.popen("lightning_app --version").read()
    assert __version__ in res


def test_main_catches_api_exceptions():
    assert isinstance(_main, _ApiExceptionHandler)

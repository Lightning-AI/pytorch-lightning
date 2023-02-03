import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from lightning.app import __version__
from lightning.app.cli.lightning_cli import _main, login, logout, run
from lightning.app.cli.lightning_cli_create import create, create_cluster
from lightning.app.cli.lightning_cli_delete import delete, delete_cluster
from lightning.app.cli.lightning_cli_list import get_list, list_apps, list_clusters
from lightning.app.utilities.exceptions import _ApiExceptionHandler


@pytest.mark.parametrize("command", [_main, run, get_list, create, delete])
def test_commands(command):
    runner = CliRunner()
    result = runner.invoke(command)
    assert result.exit_code == 0


def test_main_lightning_cli_no_arguments():
    """Validate the Lightning CLI without args."""
    res = os.popen("lightning").read()
    assert "login   " in res
    assert "logout  " in res
    assert "run     " in res
    assert "list    " in res
    assert "delete  " in res
    assert "create  " in res
    assert "show    " in res
    assert "ssh     " in res


def test_main_lightning_cli_help():
    """Validate the Lightning CLI."""
    res = os.popen("lightning --help").read()
    assert "login   " in res
    assert "logout  " in res
    assert "run     " in res
    assert "list    " in res
    assert "delete  " in res
    assert "create  " in res
    assert "show    " in res
    assert "ssh     " in res

    res = os.popen("lightning run --help").read()
    assert "app  " in res

    # hidden run commands should not appear in the help text
    assert "server" not in res
    assert "flow" not in res
    assert "work" not in res
    assert "frontend" not in res

    # inspect show group
    res = os.popen("lightning show --help").read()
    assert "logs " in res
    assert "cluster " in res

    # inspect show cluster group
    res = os.popen("lightning show cluster --help").read()
    assert "logs " in res

    # inspect create group
    res = os.popen("lightning create --help").read()
    assert "cluster " in res
    assert "ssh-key " in res

    # inspect delete group
    res = os.popen("lightning delete --help").read()
    assert "cluster " in res
    assert "ssh-key " in res


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.cli.cmd_clusters.AWSClusterManager.create")
def test_create_cluster(create_command: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(
        create_cluster,
        [
            "test-7",
            "--provider",
            "aws",
            "--external-id",
            "dummy",
            "--role-arn",
            "arn:aws:iam::1234567890:role/lai-byoc",
            "--sync",
        ],
    )

    create_command.assert_called_once_with(
        cluster_id="test-7",
        region="us-east-1",
        role_arn="arn:aws:iam::1234567890:role/lai-byoc",
        external_id="dummy",
        edit_before_creation=False,
        cost_savings=True,
        do_async=False,
    )


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.cli.cmd_apps._AppManager.list")
def test_list_apps(list_command: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(list_apps)

    list_command.assert_called_once_with(cluster_id=None)


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.cli.cmd_clusters.AWSClusterManager.list")
def test_list_clusters(list_command: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(list_clusters)

    list_command.assert_called_once_with()


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning.app.cli.cmd_clusters.AWSClusterManager.delete")
def test_delete_cluster(delete: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(delete_cluster, ["test-7", "--sync"])

    delete.assert_called_once_with(cluster_id="test-7", force=False, do_async=False)


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
    res = os.popen("lightning --version").read()
    assert __version__ in res


def test_main_catches_api_exceptions():
    assert isinstance(_main, _ApiExceptionHandler)

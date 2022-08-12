import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from lightning_cloud.openapi import Externalv1LightningappInstance

from lightning_app.cli.lightning_cli import _main, get_app_url, login, logout, run
from lightning_app.cli.lightning_cli_create import create, create_cluster
from lightning_app.cli.lightning_cli_delete import delete, delete_cluster
from lightning_app.cli.lightning_cli_list import get_list, list_apps, list_clusters
from lightning_app.runners.runtime_type import RuntimeType


@pytest.mark.parametrize(
    "runtime_type, extra_args, lightning_cloud_url, expected_url",
    [
        (
            RuntimeType.CLOUD,
            (Externalv1LightningappInstance(id="test-app-id"),),
            "https://b975913c4b22eca5f0f9e8eff4c4b1c315340a0d.staging.lightning.ai",
            "https://b975913c4b22eca5f0f9e8eff4c4b1c315340a0d.staging.lightning.ai/me/apps/test-app-id",
        ),
        (
            RuntimeType.CLOUD,
            (Externalv1LightningappInstance(id="test-app-id"),),
            "http://localhost:9800",
            "http://localhost:9800/me/apps/test-app-id",
        ),
        (RuntimeType.SINGLEPROCESS, tuple(), "", "http://127.0.0.1:7501/view"),
        (RuntimeType.SINGLEPROCESS, tuple(), "http://localhost:9800", "http://127.0.0.1:7501/view"),
        (RuntimeType.MULTIPROCESS, tuple(), "", "http://127.0.0.1:7501/view"),
        (RuntimeType.MULTIPROCESS, tuple(), "http://localhost:9800", "http://127.0.0.1:7501/view"),
    ],
)
def test_start_target_url(runtime_type, extra_args, lightning_cloud_url, expected_url):
    with mock.patch(
        "lightning_app.cli.lightning_cli.get_lightning_cloud_url", mock.MagicMock(return_value=lightning_cloud_url)
    ):
        assert get_app_url(runtime_type, *extra_args) == expected_url


@pytest.mark.parametrize("command", [_main, run, get_list, create, delete])
def test_commands(command):
    runner = CliRunner()
    result = runner.invoke(command)
    assert result.exit_code == 0


def test_main_lightning_cli_help():
    """Validate the Lightning CLI."""
    res = os.popen("python -m lightning --help").read()
    assert "login   " in res
    assert "logout  " in res
    assert "run     " in res
    assert "list    " in res
    assert "delete  " in res
    assert "create  " in res

    res = os.popen("python -m lightning run --help").read()
    assert "app  " in res

    # hidden run commands should not appear in the help text
    assert "server" not in res
    assert "flow" not in res
    assert "work" not in res
    assert "frontend" not in res


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.cli.cmd_clusters.AWSClusterManager.create")
@pytest.mark.parametrize(
    "extra_arguments,expected_instance_types,expected_cost_savings_mode",
    [
        (["--instance-types", "t3.xlarge"], ["t3.xlarge"], True),
        (["--instance-types", "t3.xlarge,t3.2xlarge"], ["t3.xlarge", "t3.2xlarge"], True),
        ([], None, True),
        (["--enable-performance"], None, False),
    ],
)
def test_create_cluster(
    create_command: mock.MagicMock, extra_arguments, expected_instance_types, expected_cost_savings_mode
):
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
        ]
        + extra_arguments,
    )

    create_command.assert_called_once_with(
        cluster_name="test-7",
        region="us-east-1",
        role_arn="arn:aws:iam::1234567890:role/lai-byoc",
        external_id="dummy",
        instance_types=expected_instance_types,
        edit_before_creation=False,
        cost_savings=expected_cost_savings_mode,
        wait=False,
    )


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.cli.cmd_apps._AppManager.list")
def test_list_apps(list_command: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(list_apps)

    list_command.assert_called_once_with(cluster_id=None)


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.cli.cmd_clusters.AWSClusterManager.list")
def test_list_clusters(list_command: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(list_clusters)

    list_command.assert_called_once_with()


@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.cli.cmd_clusters.AWSClusterManager.delete")
def test_delete_cluster(delete: mock.MagicMock):
    runner = CliRunner()
    runner.invoke(delete_cluster, ["test-7"])

    delete.assert_called_once_with(cluster_id="test-7", force=False, wait=False)


@mock.patch("lightning_app.utilities.login.Auth._run_server")
@mock.patch("lightning_app.utilities.login.Auth.clear")
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

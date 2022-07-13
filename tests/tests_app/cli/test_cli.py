import os
from unittest import mock

import pytest
from click.testing import CliRunner
from lightning_cloud.openapi import Externalv1LightningappInstance

from lightning_app.cli.lightning_cli import get_app_url, login, logout, main, run
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


@pytest.mark.parametrize("command", [main, run])
def test_commands(command):
    runner = CliRunner()
    result = runner.invoke(command)
    assert result.exit_code == 0


def test_main_lightning_cli_help():
    """Validate the Lightning CLI."""
    res = os.popen("python -m lightning_app --help").read()
    assert "login   " in res
    assert "logout  " in res
    assert "run     " in res

    res = os.popen("python -m lightning_app run --help").read()
    assert "app  " in res

    # hidden run commands should not appear in the help text
    assert "server" not in res
    assert "flow" not in res
    assert "work" not in res
    assert "frontend" not in res


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

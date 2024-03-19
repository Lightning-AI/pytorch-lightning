import os
from unittest import mock

import pytest
import requests
from lightning.app.utilities import login

LIGHTNING_CLOUD_URL = os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")


@pytest.fixture(autouse=True)
def before_each():
    for key in login.Keys:
        os.environ.pop(key.value, None)
    login.Auth().clear()


class TestAuthentication:
    def test_can_store_credentials(self):
        auth = login.Auth()
        auth.save(username="superman", user_id="kr-1234")
        assert auth.secrets_file.exists()

        auth.clear()
        assert not auth.secrets_file.exists()

    def test_e2e(self):
        auth = login.Auth()
        auth.save(username="superman", user_id="kr-1234")
        assert auth.secrets_file.exists()

    def test_get_auth_header_should_raise_error(self):
        with pytest.raises(AttributeError):
            login.Auth().auth_header

    def test_credentials_file_io(self):
        auth = login.Auth()
        assert not auth.secrets_file.exists()
        assert auth.load() is False
        auth.save(username="", user_id="123", api_key="123")
        assert auth.secrets_file.exists()
        assert auth.load() is True

    def test_auth_header(self):
        # fake credentials
        os.environ.setdefault("LIGHTNING_USER_ID", "7c8455e3-7c5f-4697-8a6d-105971d6b9bd")
        os.environ.setdefault("LIGHTNING_API_KEY", "e63fae57-2b50-498b-bc46-d6204cbf330e")
        auth = login.Auth()
        auth.clear()
        auth.authenticate()

        assert "Basic" in auth.auth_header
        assert (
            auth.auth_header
            == "Basic N2M4NDU1ZTMtN2M1Zi00Njk3LThhNmQtMTA1OTcxZDZiOWJkOmU2M2ZhZTU3LTJiNTAtNDk4Yi1iYzQ2LWQ2MjA0Y2JmMzMwZQ=="  # noqa E501
        )


def test_authentication_with_invalid_environment_vars():
    # if api key is passed without user id
    os.environ.setdefault("LIGHTNING_API_KEY", "123")
    with pytest.raises(ValueError):
        auth = login.Auth()
        auth.clear()
        auth.authenticate()


@mock.patch("lightning.app.utilities.login.AuthServer.login_with_browser")
def test_authentication_with_environment_vars(browser_login: mock.MagicMock):
    os.environ.setdefault("LIGHTNING_USER_ID", "abc")
    os.environ.setdefault("LIGHTNING_API_KEY", "abc")

    auth = login.Auth()
    auth.clear()
    auth.authenticate()

    assert auth.user_id == "abc"
    assert auth.auth_header == "Basic YWJjOmFiYw=="
    assert auth.authenticate() == auth.auth_header
    # should not run login flow when env vars are passed
    browser_login.assert_not_called()

    # Check credentials file
    assert auth.secrets_file.exists()
    assert auth.load() is True


def test_get_auth_url():
    auth_url = login.AuthServer().get_auth_url(1234)
    assert (
        auth_url == f"{LIGHTNING_CLOUD_URL}/sign-in?redirectTo=http%3A%2F%2Flocalhost%3A1234%2Flogin-complete"
    )  # E501


@mock.patch("lightning.app.utilities.login.find_free_network_port")
@mock.patch("uvicorn.Server.run")
@mock.patch("requests.head")
@mock.patch("click.launch")
def test_login_with_browser(
    click_launch: mock.MagicMock, head: mock.MagicMock, run: mock.MagicMock, port: mock.MagicMock
):
    port.return_value = 1234
    login.Auth()._run_server()
    url = f"{LIGHTNING_CLOUD_URL}/sign-in?redirectTo=http%3A%2F%2Flocalhost%3A1234%2Flogin-complete"  # E501
    head.assert_called_once_with(url)
    click_launch.assert_called_once_with(url)
    run.assert_called_once()


@mock.patch("lightning.app.utilities.login.find_free_network_port")
@mock.patch("uvicorn.Server.run")
@mock.patch("requests.head")
@mock.patch("click.launch")
def test_authenticate(click_launch: mock.MagicMock, head: mock.MagicMock, run: mock.MagicMock, port: mock.MagicMock):
    port.return_value = 1234
    auth = login.Auth()
    auth.clear()

    click_launch.side_effect = lambda _: auth.save("", "user_id", "api_key", "user_id")

    auth.authenticate()
    url = f"{LIGHTNING_CLOUD_URL}/sign-in?redirectTo=http%3A%2F%2Flocalhost%3A1234%2Flogin-complete"  # E501
    head.assert_called_with(url)
    click_launch.assert_called_with(url)
    run.assert_called()

    assert auth.auth_header == "Basic dXNlcl9pZDphcGlfa2V5"

    auth.authenticate()
    assert auth.auth_header == "Basic dXNlcl9pZDphcGlfa2V5"


@mock.patch("uvicorn.Server.run")
@mock.patch("requests.head")
def test_network_failure(
    head: mock.MagicMock,
    run: mock.MagicMock,
):
    head.side_effect = requests.ConnectionError()
    with pytest.raises(requests.ConnectionError):
        login.Auth()._run_server()
        run.assert_not_called()

    head.side_effect = requests.RequestException()
    with pytest.raises(requests.RequestException):
        login.Auth()._run_server()
        run.assert_not_called()


def test_with_api_key_only():
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")
    hash_ = "N2M4NDU1ZTMtN2M1Zi00Njk3LThhNmQtMTA1OTcxZDZiOWJkOmU2M2ZhZTU3LTJiNTAtNDk4Yi1iYzQ2LWQ2MjA0Y2JmMzMwZQ"
    assert auth.authenticate() == f"Basic {hash_}=="  # E501

from json import dumps
from unittest.mock import MagicMock

import pytest
from click import ClickException, group
from click.testing import CliRunner
from lightning.app.utilities.exceptions import _ApiExceptionHandler
from lightning_cloud.openapi.rest import ApiException
from urllib3 import HTTPResponse


@pytest.fixture()
def mock_api_handled_group():
    @group(cls=_ApiExceptionHandler)
    def g():
        pass

    return g


@pytest.fixture()
def mock_subcommand(mock_api_handled_group):
    @mock_api_handled_group.command()
    def cmd():
        pass

    return cmd


@pytest.fixture()
def api_error_msg():
    return "This is an internal error message"


class Test_ApiExceptionHandler:
    def test_4xx_exceptions_caught_in_subcommands(self, mock_api_handled_group, mock_subcommand, api_error_msg):
        mock_subcommand.invoke = MagicMock(
            side_effect=ApiException(
                http_resp=HTTPResponse(
                    status=400,
                    reason="Bad Request",
                    body=dumps(
                        {
                            "code": 3,
                            "message": api_error_msg,
                            "details": [],
                        },
                    ),
                )
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            mock_api_handled_group,
            [mock_subcommand.name],
            standalone_mode=False,  # stop runner from raising SystemExit on ClickException
        )

        mock_subcommand.invoke.assert_called
        assert result.exit_code == 1
        assert type(result.exception) is ClickException
        assert api_error_msg == str(result.exception)

    def test_original_thrown_if_cannot_decode_body(self, mock_api_handled_group, mock_subcommand):
        mock_subcommand.invoke = MagicMock(
            side_effect=ApiException(
                http_resp=HTTPResponse(
                    status=400,
                    reason="Bad Request",
                    body="message from server is not json encoded!",
                )
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            mock_api_handled_group,
            [mock_subcommand.name],
        )

        mock_subcommand.invoke.assert_called
        assert result.exit_code == 1
        assert type(result.exception) is ApiException

import pytest

from lightning_app.utilities.cli_helpers import _format_input_env_variables


def test_format_input_env_variables():
    with pytest.raises(Exception, match="Invalid format of environment variable"):
        _format_input_env_variables(("invalid-env",))

    with pytest.raises(Exception, match="Invalid format of environment variable"):
        _format_input_env_variables(("=invalid",))

    with pytest.raises(Exception, match="Invalid format of environment variable"):
        _format_input_env_variables(("=invalid=",))

    with pytest.raises(Exception, match="is duplicated. Please only include it once."):
        _format_input_env_variables(
            (
                "FOO=bar",
                "FOO=bar",
            )
        )

    with pytest.raises(
        Exception,
        match="is not a valid name. It is only allowed to contain digits 0-9, letters A-Z",
    ):
        _format_input_env_variables(("*FOO#=bar",))

    assert _format_input_env_variables(("FOO=bar", "BLA=bloz")) == {"FOO": "bar", "BLA": "bloz"}

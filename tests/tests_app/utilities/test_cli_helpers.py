from unittest.mock import Mock

import arrow
import pytest

from lightning_app.utilities.cli_helpers import _arrow_time_callback, _format_input_env_variables


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


def test_arrow_time_callback():
    # Check ISO 8601 variations
    assert _arrow_time_callback(Mock(), Mock(), "2022.08.23") == arrow.Arrow(2022, 8, 23)

    assert _arrow_time_callback(Mock(), Mock(), "2022.08.23 12:34") == arrow.Arrow(2022, 8, 23, 12, 34)

    assert _arrow_time_callback(Mock(), Mock(), "2022-08-23 12:34") == arrow.Arrow(2022, 8, 23, 12, 34)

    assert _arrow_time_callback(Mock(), Mock(), "2022-08-23 12:34:00.000") == arrow.Arrow(2022, 8, 23, 12, 34)

    # Just check humanized format is parsed
    assert type(_arrow_time_callback(Mock(), Mock(), "48 hours ago")) == arrow.Arrow

    assert type(_arrow_time_callback(Mock(), Mock(), "60 minutes ago")) == arrow.Arrow

    assert type(_arrow_time_callback(Mock(), Mock(), "120 seconds ago")) == arrow.Arrow

    # Check raising errors
    with pytest.raises(Exception, match="cannot parse time Mon"):
        _arrow_time_callback(Mock(), Mock(), "Mon")

    with pytest.raises(Exception, match="cannot parse time Mon Sep 08 16:41:45 2022"):
        _arrow_time_callback(Mock(), Mock(), "Mon Sep 08 16:41:45 2022")

    with pytest.raises(Exception, match="cannot parse time 2022.125.12"):
        _arrow_time_callback(Mock(), Mock(), "2022.125.12")

    with pytest.raises(Exception, match="cannot parse time 1 time unit ago"):
        _arrow_time_callback(Mock(), Mock(), "1 time unit ago")

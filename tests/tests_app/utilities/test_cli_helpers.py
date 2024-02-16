import os
import sys
from unittest.mock import Mock, patch

import arrow
import lightning.app
import pytest
from lightning.app.utilities.cli_helpers import (
    _arrow_time_callback,
    _check_environment_and_redirect,
    _format_input_env_variables,
    _get_newer_version,
)


def test_format_input_env_variables():
    with pytest.raises(Exception, match="Invalid format of environment variable"):
        _format_input_env_variables(("invalid-env",))

    with pytest.raises(Exception, match="Invalid format of environment variable"):
        _format_input_env_variables(("=invalid",))

    with pytest.raises(Exception, match="Invalid format of environment variable"):
        _format_input_env_variables(("=invalid=",))

    with pytest.raises(Exception, match="is duplicated. Please only include it once."):
        _format_input_env_variables((
            "FOO=bar",
            "FOO=bar",
        ))

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
    assert type(_arrow_time_callback(Mock(), Mock(), "48 hours ago")) is arrow.Arrow

    assert type(_arrow_time_callback(Mock(), Mock(), "60 minutes ago")) is arrow.Arrow

    assert type(_arrow_time_callback(Mock(), Mock(), "120 seconds ago")) is arrow.Arrow

    # Check raising errors
    with pytest.raises(Exception, match="cannot parse time Mon"):
        _arrow_time_callback(Mock(), Mock(), "Mon")

    with pytest.raises(Exception, match="cannot parse time Mon Sep 08 16:41:45 2022"):
        _arrow_time_callback(Mock(), Mock(), "Mon Sep 08 16:41:45 2022")

    with pytest.raises(Exception, match="cannot parse time 2022.125.12"):
        _arrow_time_callback(Mock(), Mock(), "2022.125.12")

    with pytest.raises(Exception, match="cannot parse time 1 time unit ago"):
        _arrow_time_callback(Mock(), Mock(), "1 time unit ago")


@pytest.mark.parametrize(
    ("response", "current_version", "newer_version"),
    [
        (
            {
                "info": {
                    "version": "2.0.0",
                    "yanked": False,
                },
                "releases": {
                    "1.0.0": {},
                    "2.0.0": {},
                },
            },
            "1.0.0",
            "2.0.0",
        ),
        (
            {
                "info": {
                    "version": "2.0.0",
                    "yanked": True,
                },
                "releases": {
                    "1.0.0": {},
                    "2.0.0": {},
                },
            },
            "1.0.0",
            None,
        ),
        (
            {
                "info": {
                    "version": "1.0.0",
                    "yanked": False,
                },
                "releases": {
                    "1.0.0": {},
                },
            },
            "1.0.0",
            None,
        ),
        (
            {
                "info": {
                    "version": "2.0.0rc0",
                    "yanked": False,
                },
                "releases": {
                    "1.0.0": {},
                    "2.0.0": {},
                },
            },
            "1.0.0",
            None,
        ),
        (
            {
                "info": {
                    "version": "2.0.0",
                    "yanked": False,
                },
                "releases": {
                    "1.0.0": {},
                    "2.0.0": {},
                },
            },
            "1.0.0dev",
            None,
        ),
        ({"this wil trigger an error": True}, "1.0.0", None),
        ({}, "1.0.0rc0", None),
    ],
)
@patch("lightning.app.utilities.cli_helpers.requests")
def test_get_newer_version(mock_requests, response, current_version, newer_version):
    mock_requests.get().json.return_value = response

    lightning.app.utilities.cli_helpers.__version__ = current_version

    _get_newer_version.cache_clear()
    assert _get_newer_version() == newer_version


@patch("lightning.app.utilities.cli_helpers._redirect_command")
def test_check_environment_and_redirect(mock_redirect_command, tmpdir, monkeypatch):
    # Ensure that the test fails if it tries to redirect
    mock_redirect_command.side_effect = RuntimeError

    # Test normal executable on the path
    # Ensure current executable is on the path
    monkeypatch.setenv("PATH", f"{os.path.dirname(sys.executable)}")

    assert _check_environment_and_redirect() is None

    # Test executable on the path with redirect
    fake_python_path = os.path.join(tmpdir, "python")

    os.symlink(sys.executable, fake_python_path)

    monkeypatch.setenv("PATH", f"{tmpdir}")
    assert _check_environment_and_redirect() is None

    os.remove(fake_python_path)

    descriptor = os.open(
        fake_python_path,
        flags=(
            os.O_WRONLY  # access mode: write only
            | os.O_CREAT  # create if not exists
            | os.O_TRUNC  # truncate the file to zero
        ),
        mode=0o777,
    )

    with open(descriptor, "w") as f:
        f.writelines([
            "#!/bin/bash\n",
            f'{sys.executable} "$@"',
        ])

    monkeypatch.setenv("PATH", f"{tmpdir}")
    assert _check_environment_and_redirect() is None

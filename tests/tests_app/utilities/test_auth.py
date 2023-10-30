from typing import Dict

import pytest
from lightning.app.utilities.auth import _credential_string_to_basic_auth_params


@pytest.mark.parametrize(
    ("credential_string", "expected_parsed", "exception_message"),
    [
        ("", None, "Credential string must follow the format username:password; the provided one ('') does not."),
        (":", None, "Username cannot be empty."),
        (":pass", None, "Username cannot be empty."),
        ("user:", None, "Password cannot be empty."),
        ("user:pass", {"username": "user", "password": "pass"}, ""),
    ],
)
def test__credential_string_to_basic_auth_params(
    credential_string: str, expected_parsed: Dict[str, str], exception_message: str
):
    if expected_parsed is not None:
        assert _credential_string_to_basic_auth_params(credential_string) == expected_parsed
    else:
        with pytest.raises(ValueError) as exception:
            _credential_string_to_basic_auth_params(credential_string)
        assert exception_message == str(exception.value)

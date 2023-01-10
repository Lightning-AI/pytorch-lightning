from typing import Dict


def _credential_string_to_basic_auth_params(credential_string: str) -> Dict[str, str]:
    """Returns the name/ID pair for each given Secret name.

    Raises a `ValueError` if any of the given Secret names do not exist.
    """
    if credential_string.count(":") != 1:
        raise ValueError(
            "Credential string must follow the format username:password; "
            + f"the provided one ('{credential_string}') does not."
        )

    username, password = credential_string.split(":")

    if not username:
        raise ValueError("Username cannot be empty.")

    if not password:
        raise ValueError("Password cannot be empty.")

    return {"username": username, "password": password}

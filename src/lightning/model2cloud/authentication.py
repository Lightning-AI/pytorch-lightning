import json
import webbrowser

import requests
from requests.models import HTTPBasicAuth

from lightning.app.model2cloud.utils import LIGHTNING_CLOUD_URL
from lightning.app.utilities.network import LightningClient


def get_user_details():
    def _get_user_details():
        client = LightningClient()
        user_details = client.auth_service_get_user()
        return (user_details.username, user_details.api_key)

    username, api_key = _get_user_details()
    return username, api_key


def get_username_from_api_key(api_key: str):
    response = requests.get(
        url=f"{LIGHTNING_CLOUD_URL}/v1/auth/user",
        auth=HTTPBasicAuth("lightning", api_key),
    )
    assert response.status_code == 200, (
        "API_KEY provided is either invalid or wasn't found in the database."
        " Please ensure that you passed the correct API_KEY."
    )
    return json.loads(response.content)["username"]


def _check_browser_runnable():
    try:
        webbrowser.get()
        return True
    except webbrowser.Error:
        return False


def authenticate(inp_api_key: str = ""):
    if not inp_api_key:
        if not _check_browser_runnable():
            raise ValueError(
                "Couldn't find a runnable browser in the current system/server."
                " In order to run the commands on this system, we suggest passing the `api_key`"
                " after logging into https://lightning.ai."
            )
        username, api_key = get_user_details()
        return username, api_key

    username = get_username_from_api_key(inp_api_key)
    return username, inp_api_key

# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import webbrowser

import requests
from requests.models import HTTPBasicAuth

from lightning.app.utilities.network import LightningClient

_LIGHTNING_CLOUD_URL = os.getenv("LIGHTNING_CLOUD_URL", default="https://lightning.ai")


def get_user_details():
    def _get_user_details():
        client = LightningClient()
        user_details = client.auth_service_get_user()
        return (user_details.username, user_details.api_key)

    username, api_key = _get_user_details()
    return username, api_key


def get_username_from_api_key(api_key: str):
    response = requests.get(url=f"{_LIGHTNING_CLOUD_URL}/v1/auth/user", auth=HTTPBasicAuth("lightning", api_key))
    if response.status_code != 200:
        raise ConnectionRefusedError(
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

# Copyright The Lightning AI team.
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

import base64
import json
import os
import pathlib
from dataclasses import dataclass
from enum import Enum
from time import sleep
from typing import Optional
from urllib.parse import urlencode

import click
import requests
import uvicorn
from fastapi import FastAPI, Query, Request
from starlette.background import BackgroundTask
from starlette.responses import RedirectResponse

from lightning.app.core.constants import get_lightning_cloud_url, LIGHTNING_CREDENTIAL_PATH
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.network import find_free_network_port

logger = Logger(__name__)


class Keys(Enum):
    USERNAME = "LIGHTNING_USERNAME"
    USER_ID = "LIGHTNING_USER_ID"
    API_KEY = "LIGHTNING_API_KEY"

    @property
    def suffix(self):
        return self.value.lstrip("LIGHTNING_").lower()


@dataclass
class Auth:
    username: Optional[str] = None
    user_id: Optional[str] = None
    api_key: Optional[str] = None

    secrets_file = pathlib.Path(LIGHTNING_CREDENTIAL_PATH)

    def load(self) -> bool:
        """Load credentials from disk and update properties with credentials.

        Returns
        ----------
        True if credentials are available.
        """
        if not self.secrets_file.exists():
            logger.debug("Credentials file not found.")
            return False
        with self.secrets_file.open() as creds:
            credentials = json.load(creds)
            for key in Keys:
                setattr(self, key.suffix, credentials.get(key.suffix, None))
            return True

    def save(self, token: str = "", user_id: str = "", api_key: str = "", username: str = "") -> None:
        """save credentials to disk."""
        self.secrets_file.parent.mkdir(exist_ok=True, parents=True)
        with self.secrets_file.open("w") as f:
            json.dump(
                {
                    f"{Keys.USERNAME.suffix}": username,
                    f"{Keys.USER_ID.suffix}": user_id,
                    f"{Keys.API_KEY.suffix}": api_key,
                },
                f,
            )

        self.username = username
        self.user_id = user_id
        self.api_key = api_key
        logger.debug("credentials saved successfully")

    def clear(self) -> None:
        """Remove credentials from disk."""
        if self.secrets_file.exists():
            self.secrets_file.unlink()
        for key in Keys:
            setattr(self, key.suffix, None)
        logger.debug("credentials removed successfully")

    @property
    def auth_header(self) -> Optional[str]:
        """authentication header used by lightning-cloud client."""
        if self.api_key:
            token = f"{self.user_id}:{self.api_key}"
            return f"Basic {base64.b64encode(token.encode('ascii')).decode('ascii')}"  # E501
        raise AttributeError(
            "Authentication Failed, no authentication header available. "
            "This is most likely a bug in the LightningCloud Framework"
        )

    def _run_server(self) -> None:
        """start a server to complete authentication."""
        AuthServer().login_with_browser(self)

    def authenticate(self) -> Optional[str]:
        """Performs end to end authentication flow.

        Returns
        ----------
        authorization header to use when authentication completes.
        """
        if not self.load():
            # First try to authenticate from env
            for key in Keys:
                setattr(self, key.suffix, os.environ.get(key.value, None))

            if self.user_id and self.api_key:
                self.save("", self.user_id, self.api_key, self.user_id)
                logger.info("Credentials loaded from environment variables")
                return self.auth_header
            elif self.api_key or self.user_id:
                raise ValueError(
                    "To use env vars for authentication both "
                    f"{Keys.USER_ID.value} and {Keys.API_KEY.value} should be set."
                )

            logger.debug("failed to load credentials, opening browser to get new.")
            self._run_server()
            return self.auth_header

        elif self.user_id and self.api_key:
            return self.auth_header

        raise ValueError(
            "We couldn't find any credentials linked to your account. "
            "Please try logging in using the CLI command `lightning login`"
        )


class AuthServer:
    @staticmethod
    def get_auth_url(port: int) -> str:
        redirect_uri = f"http://localhost:{port}/login-complete"
        params = urlencode(dict(redirectTo=redirect_uri))
        return f"{get_lightning_cloud_url()}/sign-in?{params}"

    def login_with_browser(self, auth: Auth) -> None:
        app = FastAPI()
        port = find_free_network_port()
        url = self.get_auth_url(port)

        try:
            # check if server is reachable or catch any network errors
            requests.head(url)
        except requests.ConnectionError as e:
            raise requests.ConnectionError(
                f"No internet connection available. Please connect to a stable internet connection \n{e}"  # E501
            )
        except requests.RequestException as e:
            raise requests.RequestException(
                f"An error occurred with the request. Please report this issue to Lightning Team \n{e}"  # E501
            )

        logger.info(
            "\nAttempting to automatically open the login page in your default browser.\n"
            'If the browser does not open, navigate to the "Keys" tab on your Lightning AI profile page:\n\n'
            f"{get_lightning_cloud_url()}/me/keys\n\n"
            'Copy the "Headless CLI Login" command, and execute it in your terminal.\n'
        )
        click.launch(url)

        @app.get("/login-complete")
        async def save_token(request: Request, token="", key="", user_id: str = Query("", alias="userID")):
            async def stop_server_once_request_is_done():
                while not await request.is_disconnected():
                    sleep(0.25)
                server.should_exit = True

            if not token:
                logger.warn(
                    "Login Failed. This is most likely because you're using an older version of the CLI. \n"  # noqa E501
                    "Please try to update the CLI or open an issue with this information \n"  # E501
                    f"expected token in {request.query_params.items()}"
                )
                return RedirectResponse(
                    url=f"{get_lightning_cloud_url()}/cli-login-failed",
                    background=BackgroundTask(stop_server_once_request_is_done),
                )

            auth.save(token=token, username=user_id, user_id=user_id, api_key=key)
            logger.info("Login Successful")

            # Include the credentials in the redirect so that UI will also be logged in
            params = urlencode(dict(token=token, key=key, userID=user_id))

            return RedirectResponse(
                url=f"{get_lightning_cloud_url()}/cli-login-successful?{params}",
                background=BackgroundTask(stop_server_once_request_is_done),
            )

        server = uvicorn.Server(config=uvicorn.Config(app, port=port, log_level="error"))
        server.run()

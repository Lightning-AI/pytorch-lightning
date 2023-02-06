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

from json import JSONDecodeError, loads
from typing import Any

from click import ClickException, Context, Group
from lightning_cloud.openapi.rest import ApiException


class _ApiExceptionHandler(Group):
    """Attempts to convert ApiExceptions to ClickExceptions.

    This process clarifies the error for the user by:
    1. Showing the error message from the lightning.ai servers,
       instead of showing the entire HTTP response
    2. Suppressing long tracebacks

    However, if the ApiException cannot be decoded, or is not
    a 4xx error, the original ApiException will be re-raised.
    """

    def invoke(self, ctx: Context) -> Any:
        try:
            return super().invoke(ctx)
        except ApiException as api:
            exception_messages = []
            if 400 <= api.status < 500:
                try:
                    body = loads(api.body)
                except JSONDecodeError:
                    raise api
                exception_messages.append(body["message"])
                exception_messages.extend(body["details"])
            else:
                raise api
            raise ClickException("\n".join(exception_messages))


class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with Lightning."""


class CacheMissException(Exception):
    """Exception used internally as a boundary to non-executed functions."""


class ExitAppException(Exception):
    """Exception used by components to signal that App should exit."""


class LightningComponentException(Exception):
    """Exception used to inform users of misuse with LightningComponent."""


class InvalidPathException(Exception):
    """Exception used to inform users they are accessing an invalid path."""


class LightningFlowException(Exception):
    """Exception used to inform users of misuse with LightningFlow."""


class LightningWorkException(Exception):
    """Exception used to inform users of misuse with LightningWork."""


class LightningPlatformException(Exception):  # pragma: no cover
    """Exception used to inform users of issues related to platform the LightningApp is running on.

    It gets raised by the Lightning Launcher on the platform side when the app is running in the cloud, and is useful
    when framework or user code needs to catch exceptions specific to the platform, e.g., when resources exceed quotas.
    """


class LightningAppStateException(Exception):
    """Exception to inform users of app state errors."""


class LightningSigtermStateException(Exception):
    """Exception to propagate exception in work proxy."""

    def __init__(self, exit_code):
        self.exit_code = exit_code


class LogLinesLimitExceeded(Exception):
    """Exception to inform the user that we've reached the maximum number of log lines."""

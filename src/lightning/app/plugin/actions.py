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
from dataclasses import dataclass
from enum import Enum
from typing import Union

from lightning_cloud.openapi.models import V1CloudSpaceAppAction, V1CloudSpaceAppActionType


class _Action:
    """Actions are returned by `LightningPlugin` objects to perform actions in the UI."""

    def to_spec(self) -> V1CloudSpaceAppAction:
        """Convert this action to a ``V1CloudSpaceAppAction``"""
        raise NotImplementedError


@dataclass
class NavigateTo(_Action):
    """The ``NavigateTo`` action can be used to navigate to a relative URL within the Lightning frontend.

    Args:
        url: The relative URL to navigate to. E.g. ``/<username>/<project>``.
    """

    url: str

    def to_spec(self) -> V1CloudSpaceAppAction:
        return V1CloudSpaceAppAction(
            type=V1CloudSpaceAppActionType.NAVIGATE_TO,
            content=self.url,
        )


class ToastSeverity(Enum):
    ERROR = "error"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"

    def __str__(self) -> str:
        return self.value


@dataclass
class Toast(_Action):
    """The ``Toast`` action can be used to display a toast message to the user.

    Args:
        severity: The severity level of the toast. One of: "error", "info", "success", "warning".
        message: The message body.
    """

    severity: Union[ToastSeverity, str]
    message: str

    def to_spec(self) -> V1CloudSpaceAppAction:
        return V1CloudSpaceAppAction(
            type=V1CloudSpaceAppActionType.TOAST,
            content=f"{self.severity}:{self.message}",
        )

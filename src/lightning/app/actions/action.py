from dataclasses import dataclass
from enum import Enum
from typing import Union

from lightning_cloud.openapi.models import V1CloudSpaceAppAction, V1CloudSpaceAppActionType


class Action:
    """Actions are returned by `LightningPlugin` objects to perform actions in the UI."""

    def to_spec(self) -> V1CloudSpaceAppAction:
        """Convert this action to a ``V1CloudSpaceAppAction``"""
        raise NotImplementedError


@dataclass
class NavigateTo(Action):
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

    def __str__(self):
        return self.value


@dataclass
class Toast(Action):
    severity: Union[ToastSeverity, str]
    message: str

    def to_spec(self) -> V1CloudSpaceAppAction:
        return V1CloudSpaceAppAction(
            type=V1CloudSpaceAppActionType.TOAST,
            content=f"{self.severity}:{self.message}",
        )

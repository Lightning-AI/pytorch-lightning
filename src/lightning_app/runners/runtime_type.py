from enum import Enum
from typing import Type, TYPE_CHECKING

from lightning_app.runners import CloudRuntime, MultiProcessRuntime, SingleProcessRuntime

if TYPE_CHECKING:
    from lightning_app.runners.runtime import Runtime


class RuntimeType(Enum):
    SINGLEPROCESS = "singleprocess"
    MULTIPROCESS = "multiprocess"
    CLOUD = "cloud"

    def get_runtime(self) -> Type["Runtime"]:
        if self == RuntimeType.SINGLEPROCESS:
            return SingleProcessRuntime
        elif self == RuntimeType.MULTIPROCESS:
            return MultiProcessRuntime
        elif self == RuntimeType.CLOUD:
            return CloudRuntime
        else:
            raise ValueError("Unknown runtime type")

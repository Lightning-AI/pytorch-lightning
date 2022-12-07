from enum import Enum
from typing import Type, TYPE_CHECKING

from lightning_app.runners import CloudRuntime, MultiProcessRuntime

if TYPE_CHECKING:
    from lightning_app.runners.runtime import Runtime


class RuntimeType(Enum):
    MULTIPROCESS = "multiprocess"
    CLOUD = "cloud"

    def get_runtime(self) -> Type["Runtime"]:
        if self == RuntimeType.MULTIPROCESS:
            return MultiProcessRuntime
        elif self == RuntimeType.CLOUD:
            return CloudRuntime
        else:
            raise ValueError("Unknown runtime type")

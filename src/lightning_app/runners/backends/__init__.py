from enum import Enum

from lightning_app.core.constants import APP_SERVER_IN_CLOUD
from lightning_app.runners.backends.backend import Backend
from lightning_app.runners.backends.cloud import CloudBackend
from lightning_app.runners.backends.docker import DockerBackend
from lightning_app.runners.backends.mp_process import CloudMultiProcessingBackend, MultiProcessingBackend


class BackendType(Enum):
    MULTIPROCESSING = "multiprocessing"
    DOCKER = "docker"
    CLOUD = "cloud"

    def get_backend(self, entrypoint_file: str) -> "Backend":
        if self == BackendType.MULTIPROCESSING:
            if APP_SERVER_IN_CLOUD:
                return CloudMultiProcessingBackend(entrypoint_file)
            return MultiProcessingBackend(entrypoint_file)
        elif self == BackendType.DOCKER:
            return DockerBackend(entrypoint_file)
        elif self == BackendType.CLOUD:
            return CloudBackend(entrypoint_file)
        else:
            raise ValueError("Unknown client type")

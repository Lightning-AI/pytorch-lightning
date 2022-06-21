from time import time
from typing import List

import lightning_app
from lightning_app.core.queues import QueuingSystem
from lightning_app.runners.backends.backend import Backend


class DockerBackend(Backend):
    def __init__(self, entrypoint_file: str):
        super().__init__(entrypoint_file=entrypoint_file, queues=QueuingSystem.REDIS, queue_id=str(int(time())))

    def create_work(self, app, work):
        pass

    def update_work_statuses(self, works) -> None:
        pass

    def stop_all_works(self, works: List["lightning_app.LightningWork"]) -> None:
        pass

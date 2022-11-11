from typing import List, Optional, TYPE_CHECKING

from lightning_app.core.queues import QueuingSystem
from lightning_app.runners.backends import Backend
from lightning_app.utilities.network import LightningClient

if TYPE_CHECKING:
    import lightning_app


class CloudBackend(Backend):
    def __init__(self, entrypoint_file, queue_id: Optional[str] = None, status_update_interval: int = None):
        super().__init__(entrypoint_file, queues=QueuingSystem.MULTIPROCESS, queue_id=queue_id)
        self.client = LightningClient(retry=True)

    def create_work(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork") -> None:
        raise NotImplementedError

    def update_work_statuses(self, works: List["lightning_app.LightningWork"]) -> None:
        raise NotImplementedError

    def stop_all_works(self, works: List["lightning_app.LightningWork"]) -> None:
        raise NotImplementedError

    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        raise NotImplementedError

    def stop_work(self, app: "lightning_app.LightningApp", work: "lightning_app.LightningWork") -> None:
        raise NotImplementedError

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class WorkStatus(BaseModel):
    """The ``WorkStatus`` captures the status of a work according to the app."""

    stage: str
    timestamp: float
    reason: Optional[str] = None
    message: Optional[str] = None
    count: int = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        assert self.timestamp > 0 and self.timestamp < (int(datetime.now().timestamp()) + 10)


class AppStatus(BaseModel):
    """The ``AppStatus`` captures the current status of the app and its components."""

    # ``True`` when the app UI is ready to be viewed
    is_ui_ready: bool

    # The statuses of ``LightningWork`` objects currently associated with this app
    work_statuses: Dict[str, WorkStatus]

from typing import List, Optional

from pydantic import BaseModel

from lightning_app.utilities.enum import WorkStageStatus


class WorkStatus(BaseModel):
    """The ``WorkStatus`` captures the status of a work according to the app."""

    name: str
    stage: WorkStageStatus
    timestamp: float
    reason: Optional[str] = None
    message: Optional[str] = None
    count: int = 1


class AppStatus(BaseModel):
    """The ``AppStatus`` captures the current status of the app and its components."""

    # ``True`` when the app UI is ready to be viewed
    is_ui_ready: bool

    # The statuses of ``LightningWork`` objects currently associated with this app
    work_statuses: List[WorkStatus]

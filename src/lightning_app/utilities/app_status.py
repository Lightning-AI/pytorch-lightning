from typing import List

from pydantic import BaseModel


class WorkStatus(BaseModel):
    """The ``WorkStatus`` captures the status of a work according to the app."""

    # The name of the work
    name: str

    # ``True`` when the work is running according to the app.
    # Compute states in the cloud are owned by the platform.
    is_running: bool


class AppStatus(BaseModel):
    """The ``AppStatus`` captures the current status of the app and its components."""

    # ``True`` when the app UI is ready to be viewed
    is_ui_ready: bool

    # The statuses of ``LightningWork`` objects currently associated with this app
    work_statuses: List[WorkStatus]

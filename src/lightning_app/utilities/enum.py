import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


class ComponentContext(enum.Enum):
    """Describes whether the current process is running LightningFlow or LightningWork."""

    FLOW = "flow"
    WORK = "work"
    FRONTEND = "frontend"


class AppStage(enum.Enum):
    BLOCKING = "blocking"
    RUNNING = "running"
    RESTARTING = "restarting"
    STOPPING = "stopping"
    FAILED = "failed"


class WorkFailureReasons:
    TIMEOUT = "timeout"  # triggered when pending and wait timeout has been passed
    SPOT_RETRIVAL = "spot_retrival"  # triggered when a SIGTERM signal is sent the spot instance work.
    USER_EXCEPTION = "user_exception"  # triggered when an exception is raised by user code.
    INVALID_RETURN_VALUE = "invalid_return_value"  # triggered when the return value isn't valid.


class WorkStopReasons:
    SIGTERM_SIGNAL_HANDLER = "sigterm_signal_handler"
    PENDING = "pending"


class WorkPendingReason(enum.Enum):
    IMAGE_BUILDING = "image_building"
    REQUESTING_RESOURCE = "requesting_ressource"


class WorkStageStatus:
    NOT_STARTED = "not_started"
    STOPPED = "stopped"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class WorkStatus:
    stage: WorkStageStatus
    timestamp: float
    reason: Optional[str] = None
    message: Optional[str] = None
    count: int = 1

    def __post_init__(self):
        assert self.timestamp > 0 and self.timestamp < (int(datetime.now().timestamp()) + 10)


def make_status(stage: str, message: Optional[str] = None, reason: Optional[str] = None):
    return {
        "stage": stage,
        "message": message,
        "reason": reason,
        "timestamp": datetime.now(tz=timezone.utc).timestamp(),
    }

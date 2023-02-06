# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
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
    STARTED = "started"
    STOPPED = "stopped"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def make_status(stage: str, message: Optional[str] = None, reason: Optional[str] = None):
    status = {
        "stage": stage,
        "timestamp": datetime.now(tz=timezone.utc).timestamp(),
    }
    if message:
        status["message"] = message
    if reason:
        status["reason"] = reason
    return status


class CacheCallsKeys:
    LATEST_CALL_HASH = "latest_call_hash"


class OpenAPITags:
    APP_CLIENT_COMMAND = "app_client_command"
    APP_COMMAND = "app_command"
    APP_API = "app_api"

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

import os
from pathlib import Path
from typing import Optional

import lightning_cloud.env


def get_lightning_cloud_url() -> str:
    # DO NOT CHANGE!
    return os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")


SUPPORTED_PRIMITIVE_TYPES = (type(None), str, int, float, bool)
STATE_UPDATE_TIMEOUT = 0.001
STATE_ACCUMULATE_WAIT = 0.15
# Duration in seconds of a moving average of a full flow execution
# beyond which an exception is raised.
FLOW_DURATION_THRESHOLD = 1.0
# Number of samples for the moving average of the duration of flow execution
FLOW_DURATION_SAMPLES = 5

APP_SERVER_HOST = os.getenv("LIGHTNING_APP_STATE_URL", "http://127.0.0.1")
APP_SERVER_IN_CLOUD = "http://lightningapp" in APP_SERVER_HOST
APP_SERVER_PORT = 7501
APP_STATE_MAX_SIZE_BYTES = 1024 * 1024  # 1 MB

WARNING_QUEUE_SIZE = 1000
# different flag because queue debug can be very noisy, and almost always not useful unless debugging the queue itself.
QUEUE_DEBUG_ENABLED = bool(int(os.getenv("LIGHTNING_QUEUE_DEBUG_ENABLED", "0")))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_QUEUES_READ_DEFAULT_TIMEOUT = 0.005

HTTP_QUEUE_URL = os.getenv("LIGHTNING_HTTP_QUEUE_URL", "http://localhost:9801")
HTTP_QUEUE_REFRESH_INTERVAL = float(os.getenv("LIGHTNING_HTTP_QUEUE_REFRESH_INTERVAL", "1"))
HTTP_QUEUE_TOKEN = os.getenv("LIGHTNING_HTTP_QUEUE_TOKEN", None)

USER_ID = os.getenv("USER_ID", "1234")
FRONTEND_DIR = str(Path(__file__).parent.parent / "ui")
PACKAGE_LIGHTNING = os.getenv("PACKAGE_LIGHTNING", None)
CLOUD_UPLOAD_WARNING = int(os.getenv("CLOUD_UPLOAD_WARNING", "2"))
DISABLE_DEPENDENCY_CACHE = bool(int(os.getenv("DISABLE_DEPENDENCY_CACHE", "0")))
# Project under which the resources need to run in cloud. If this env is not set,
# cloud runner will try to get the default project from the cloud
LIGHTNING_CLOUD_PROJECT_ID = os.getenv("LIGHTNING_CLOUD_PROJECT_ID")
LIGHTNING_CLOUD_PRINT_SPECS = os.getenv("LIGHTNING_CLOUD_PRINT_SPECS")
LIGHTNING_DIR = os.getenv("LIGHTNING_DIR", str(Path.home() / ".lightning"))
LIGHTNING_CREDENTIAL_PATH = os.getenv("LIGHTNING_CREDENTIAL_PATH", str(Path(LIGHTNING_DIR) / "credentials.json"))
DOT_IGNORE_FILENAME = ".lightningignore"
LIGHTNING_COMPONENT_PUBLIC_REGISTRY = "https://lightning.ai/v1/components"
LIGHTNING_APPS_PUBLIC_REGISTRY = "https://lightning.ai/v1/apps"

LIGHTNING_CLOUDSPACE_HOST = os.getenv("LIGHTNING_CLOUDSPACE_HOST")
LIGHTNING_CLOUDSPACE_EXPOSED_PORT_COUNT = int(os.getenv("LIGHTNING_CLOUDSPACE_EXPOSED_PORT_COUNT", "0"))

# EXPERIMENTAL: ENV VARIABLES TO ENABLE MULTIPLE WORKS IN THE SAME MACHINE
DEFAULT_NUMBER_OF_EXPOSED_PORTS = int(os.getenv("DEFAULT_NUMBER_OF_EXPOSED_PORTS", "50"))
ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER = bool(
    int(os.getenv("ENABLE_MULTIPLE_WORKS_IN_NON_DEFAULT_CONTAINER", "0"))
)  # This isn't used in the cloud yet.

# env var trigger running setup commands in the app
ENABLE_APP_COMMENT_COMMAND_EXECUTION = bool(int(os.getenv("ENABLE_APP_COMMENT_COMMAND_EXECUTION", "0")))


DEBUG: bool = lightning_cloud.env.DEBUG
DEBUG_ENABLED = bool(int(os.getenv("LIGHTNING_DEBUG", "0")))
ENABLE_PULLING_STATE_ENDPOINT = bool(int(os.getenv("ENABLE_PULLING_STATE_ENDPOINT", "1")))
ENABLE_PUSHING_STATE_ENDPOINT = ENABLE_PULLING_STATE_ENDPOINT and bool(
    int(os.getenv("ENABLE_PUSHING_STATE_ENDPOINT", "1"))
)
ENABLE_STATE_WEBSOCKET = bool(int(os.getenv("ENABLE_STATE_WEBSOCKET", "0")))
ENABLE_UPLOAD_ENDPOINT = bool(int(os.getenv("ENABLE_UPLOAD_ENDPOINT", "1")))


def enable_multiple_works_in_default_container() -> bool:
    return bool(int(os.getenv("ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", "0")))


def get_cloud_queue_type() -> Optional[str]:
    value = os.getenv("LIGHTNING_CLOUD_QUEUE_TYPE", None)
    if value is None and enable_interruptible_works():
        value = "http"
    return value


# Number of seconds to wait between filesystem checks when waiting for files in remote storage
REMOTE_STORAGE_WAIT = 0.5


# interruptible support
def enable_interruptible_works() -> bool:
    return bool(int(os.getenv("LIGHTNING_INTERRUPTIBLE_WORKS", "0")))


# Get Cluster Driver
_CLUSTER_DRIVERS = [None, "k8s", "direct"]


def get_cluster_driver() -> Optional[str]:
    value = os.getenv("LIGHTNING_CLUSTER_DRIVER", None)
    if value is None:
        if enable_interruptible_works():
            value = "direct"
        else:
            value = None
    if value not in _CLUSTER_DRIVERS:
        raise ValueError(f"Found {value} cluster driver. The value needs to be in {_CLUSTER_DRIVERS}.")
    return value

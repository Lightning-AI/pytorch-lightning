import os
from pathlib import Path

import lightning_app

SUPPORTED_PRIMITIVE_TYPES = (type(None), str, int, float, bool)
STATE_UPDATE_TIMEOUT = 0.001
STATE_ACCUMULATE_WAIT = 0.05
# Duration in seconds of a moving average of a full flow execution
# beyond which an exception is raised.
FLOW_DURATION_THRESHOLD = 1.0
# Number of samples for the moving average of the duration of flow execution
FLOW_DURATION_SAMPLES = 5

APP_SERVER_HOST = os.getenv("LIGHTNING_APP_STATE_URL", "http://127.0.0.1")
APP_SERVER_PORT = 7501
APP_STATE_MAX_SIZE_BYTES = 1024 * 1024  # 1 MB
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_QUEUES_READ_DEFAULT_TIMEOUT = 0.005
REDIS_WARNING_QUEUE_SIZE = 1000
USER_ID = os.getenv("USER_ID", "1234")
FRONTEND_DIR = os.path.join(os.path.dirname(lightning_app.__file__), "ui")
PREPARE_LIGHTING = bool(int(os.getenv("PREPARE_LIGHTING", "0")))
LOCAL_LAUNCH_ADMIN_VIEW = bool(int(os.getenv("LOCAL_LAUNCH_ADMIN_VIEW", "0")))
CLOUD_UPLOAD_WARNING = int(os.getenv("CLOUD_UPLOAD_WARNING", "2"))
DISABLE_DEPENDENCY_CACHE = bool(int(os.getenv("DISABLE_DEPENDENCY_CACHE", "0")))
# Project under which the resources need to run in cloud. If this env is not set,
# cloud runner will try to get the default project from the cloud
LIGHTNING_CLOUD_PROJECT_ID = os.getenv("LIGHTNING_CLOUD_PROJECT_ID")
LIGHTNING_CREDENTIAL_PATH = os.getenv("LIGHTNING_CREDENTIAL_PATH", str(Path.home() / ".lightning" / "credentials.json"))
DOT_IGNORE_FILENAME = ".lightningignore"


def get_lightning_cloud_url() -> str:
    # DO NOT CHANGE!
    return os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")

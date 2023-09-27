import contextlib
import os
import shutil
import signal
import threading
from datetime import datetime
from pathlib import Path
from threading import Thread

import psutil
import py
import pytest
from lightning.app.core import constants
from lightning.app.storage.path import _storage_root_dir
from lightning.app.utilities.app_helpers import _collect_child_process_pids
from lightning.app.utilities.component import _set_context
from lightning.app.utilities.packaging import cloud_compute
from lightning.app.utilities.packaging.app_config import _APP_CONFIG_FILENAME
from lightning.app.utilities.state import AppState

os.environ["LIGHTNING_DISPATCHED"] = "1"

original_method = Thread._wait_for_tstate_lock


def fn(self, *args, timeout=None, **kwargs):
    original_method(self, *args, timeout=1, **kwargs)


Thread._wait_for_tstate_lock = fn


@pytest.fixture()
def caplog(caplog):
    """Workaround for https://github.com/pytest-dev/pytest/issues/3697.

    Setting ``filterwarnings`` with pytest breaks ``caplog`` when ``not logger.propagate``.

    """
    import logging

    root_logger = logging.getLogger()
    root_propagate = root_logger.propagate
    root_logger.propagate = True

    propagation_dict = {
        name: logging.getLogger(name).propagate
        for name in logging.root.manager.loggerDict
        if name.startswith("lightning.app")
    }
    for name in propagation_dict:
        logging.getLogger(name).propagate = True

    yield caplog

    root_logger.propagate = root_propagate
    for name, propagate in propagation_dict.items():
        logging.getLogger(name).propagate = propagate

import contextlib
import os
import shutil
import threading
from subprocess import Popen

import psutil
import pytest
from lightning.app.storage.path import _storage_root_dir
from lightning.app.utilities.component import _set_context
from lightning.app.utilities.packaging import cloud_compute
from lightning.app.utilities.packaging.app_config import _APP_CONFIG_FILENAME
from lightning.app.utilities.state import AppState

from integrations_app.public import _PATH_EXAMPLES

GITHUB_APP_URLS = {
    "template_react_ui": "https://github.com/Lightning-AI/lightning-template-react.git",
}

os.environ["LIGHTNING_DISPATCHED"] = "1"


def pytest_sessionstart(*_):
    """Pytest hook that get called after the Session object has been created and before performing collection and
    entering the run test loop."""
    for name, url in GITHUB_APP_URLS.items():
        app_path = _PATH_EXAMPLES / name
        if not os.path.exists(app_path):
            Popen(["git", "clone", url, name], cwd=_PATH_EXAMPLES).wait(timeout=90)
        else:
            Popen(["git", "pull", "main"], cwd=app_path).wait(timeout=90)


def pytest_sessionfinish(session, exitstatus):
    """Pytest hook that get called after whole test run finished, right before returning the exit status to the
    system."""
    # kill all the processes and threads created by parent
    # TODO this isn't great. We should have each tests doing it's own cleanup
    current_process = psutil.Process()
    for child in current_process.children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess):
            params = child.as_dict() or {}
            cmd_lines = params.get("cmdline", [])
            # we shouldn't kill the resource tracker from multiprocessing. If we do,
            # `atexit` will throw as it uses resource tracker to try to clean up
            if cmd_lines and "resource_tracker" in cmd_lines[-1]:
                continue
            child.kill()

    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join(0)


@pytest.fixture(autouse=True)
def cleanup():
    from lightning.app.utilities.app_helpers import _LightningAppRef

    yield
    _LightningAppRef._app_instance = None
    shutil.rmtree("./storage", ignore_errors=True)
    shutil.rmtree(_storage_root_dir(), ignore_errors=True)
    shutil.rmtree("./.shared", ignore_errors=True)
    if os.path.isfile(_APP_CONFIG_FILENAME):
        os.remove(_APP_CONFIG_FILENAME)
    _set_context(None)


@pytest.fixture(autouse=True)
def clear_app_state_state_variables():
    """Resets global variables in order to prevent interference between tests."""
    yield
    import lightning.app.utilities.state

    lightning.app.utilities.state._STATE = None
    lightning.app.utilities.state._LAST_STATE = None
    AppState._MY_AFFILIATION = ()
    if hasattr(cloud_compute, "_CLOUD_COMPUTE_STORE"):
        cloud_compute._CLOUD_COMPUTE_STORE.clear()

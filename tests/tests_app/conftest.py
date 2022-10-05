import os
import shutil
from datetime import datetime
from pathlib import Path
from subprocess import Popen

import psutil
import py
import pytest
from tests_app import _PROJECT_ROOT

from lightning_app.storage.path import storage_root_dir
from lightning_app.utilities.component import _set_context
from lightning_app.utilities.packaging import cloud_compute
from lightning_app.utilities.packaging.app_config import _APP_CONFIG_FILENAME
from lightning_app.utilities.state import AppState

GITHUB_APP_URLS = {
    "template_react_ui": "https://github.com/Lightning-AI/lightning-template-react.git",
}


def pytest_sessionstart(*_):
    """Pytest hook that get called after the Session object has been created and before performing collection and
    entering the run test loop."""
    for name, url in GITHUB_APP_URLS.items():
        if not os.path.exists(os.path.join(_PROJECT_ROOT, "examples", name)):
            Popen(
                ["git", "clone", url, name],
                cwd=os.path.join(
                    _PROJECT_ROOT,
                    "examples",
                ),
            ).wait(timeout=90)
        else:
            Popen(["git", "pull", "main"], cwd=os.path.join(_PROJECT_ROOT, "examples", name)).wait(timeout=90)


def pytest_sessionfinish(session, exitstatus):
    """Pytest hook that get called after whole test run finished, right before returning the exit status to the
    system."""
    # kill all the processes and threads created by parent
    # TODO this isn't great. We should have each tests doing it's own cleanup
    current_process = psutil.Process()
    for child in current_process.children(recursive=True):
        params = child.as_dict() or {}
        cmd_lines = params.get("cmdline", [])
        # we shouldn't kill the resource tracker from multiprocessing. If we do,
        # `atexit` will throw as it uses resource tracker to try to clean up
        if cmd_lines and "resource_tracker" in cmd_lines[-1]:
            continue
        child.kill()


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    from lightning_app.utilities.app_helpers import _LightningAppRef

    yield
    _LightningAppRef._app_instance = None
    shutil.rmtree("./storage", ignore_errors=True)
    shutil.rmtree(storage_root_dir(), ignore_errors=True)
    shutil.rmtree("./.shared", ignore_errors=True)
    if os.path.isfile(_APP_CONFIG_FILENAME):
        os.remove(_APP_CONFIG_FILENAME)
    _set_context(None)


@pytest.fixture(scope="function", autouse=True)
def clear_app_state_state_variables():
    """Resets global variables in order to prevent interference between tests."""
    yield
    import lightning_app.utilities.state

    lightning_app.utilities.state._STATE = None
    lightning_app.utilities.state._LAST_STATE = None
    AppState._MY_AFFILIATION = ()
    cloud_compute._CLOUD_COMPUTE_STORE.clear()


@pytest.fixture
def another_tmpdir(tmp_path: Path) -> py.path.local:
    random_dir = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    tmp_path = os.path.join(tmp_path, random_dir)
    return py.path.local(tmp_path)

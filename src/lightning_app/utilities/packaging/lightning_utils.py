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

import functools
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from packaging.version import Version

from lightning_app import _logger, _PROJECT_ROOT, _root_logger
from lightning_app.__version__ import version
from lightning_app.core.constants import FRONTEND_DIR, PACKAGE_LIGHTNING
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.git import check_github_repository, get_dir_name

logger = Logger(__name__)


# FIXME(alecmerdler): Use GitHub release artifacts once the `lightning-ui` repo is public
LIGHTNING_FRONTEND_RELEASE_URL = "https://storage.googleapis.com/grid-packages/lightning-ui/v0.0.0/build.tar.gz"


def download_frontend(root: str = _PROJECT_ROOT):
    """Downloads an archive file for a specific release of the Lightning frontend and extracts it to the correct
    directory."""
    build_dir = "build"
    frontend_dir = pathlib.Path(FRONTEND_DIR)
    download_dir = tempfile.mkdtemp()

    shutil.rmtree(frontend_dir, ignore_errors=True)

    response = urllib.request.urlopen(LIGHTNING_FRONTEND_RELEASE_URL)

    file = tarfile.open(fileobj=response, mode="r|gz")
    file.extractall(path=download_dir)

    shutil.move(os.path.join(download_dir, build_dir), frontend_dir)
    print("The Lightning UI has successfully been downloaded!")


def _cleanup(*tar_files: str):
    for tar_file in tar_files:
        shutil.rmtree(os.path.join(_PROJECT_ROOT, "dist"), ignore_errors=True)
        os.remove(tar_file)


def _prepare_wheel(path):
    with open("log.txt", "w") as logfile:
        with subprocess.Popen(
            ["rm", "-r", "dist"], stdout=logfile, stderr=logfile, bufsize=0, close_fds=True, cwd=path
        ) as proc:
            proc.wait()

        with subprocess.Popen(
            ["python", "setup.py", "sdist"],
            stdout=logfile,
            stderr=logfile,
            bufsize=0,
            close_fds=True,
            cwd=path,
        ) as proc:
            proc.wait()

    os.remove("log.txt")


def _copy_tar(project_root, dest: Path) -> str:
    dist_dir = os.path.join(project_root, "dist")
    tar_files = os.listdir(dist_dir)
    assert len(tar_files) == 1
    tar_name = tar_files[0]
    tar_path = os.path.join(dist_dir, tar_name)
    shutil.copy(tar_path, dest)
    return tar_name


def get_dist_path_if_editable_install(project_name) -> str:
    """Is distribution an editable install - modified version from pip that
    fetches egg-info instead of egg-link"""
    for path_item in sys.path:
        if not os.path.isdir(path_item):
            continue

        egg_info = os.path.join(path_item, project_name + ".egg-info")
        if os.path.isdir(egg_info):
            return path_item
    return ""


def _prepare_lightning_wheels_and_requirements(root: Path, package_name: str = "lightning") -> Optional[Callable]:
    """This function determines if lightning is installed in editable mode (for developers) and packages the
    current lightning source along with the app.

    For normal users who install via PyPi or Conda, then this function does not do anything.
    """
    if not get_dist_path_if_editable_install(package_name):
        return

    os.environ["PACKAGE_NAME"] = "app" if package_name == "lightning" + "_app" else "lightning"

    # Packaging the Lightning codebase happens only inside the `lightning` repo.
    git_dir_name = get_dir_name() if check_github_repository() else None

    is_lightning = git_dir_name and git_dir_name == package_name

    if (PACKAGE_LIGHTNING is None and not is_lightning) or PACKAGE_LIGHTNING == "0":
        return

    download_frontend(_PROJECT_ROOT)
    _prepare_wheel(_PROJECT_ROOT)

    # todo: check why logging.info is missing in outputs
    print(f"Packaged Lightning with your application. Version: {version}")

    tar_name = _copy_tar(_PROJECT_ROOT, root)

    tar_files = [os.path.join(root, tar_name)]

    # Don't skip by default
    if (PACKAGE_LIGHTNING or is_lightning) and not bool(int(os.getenv("SKIP_LIGHTING_UTILITY_WHEELS_BUILD", "0"))):
        # building and copying launcher wheel if installed in editable mode
        launcher_project_path = get_dist_path_if_editable_install("lightning_launcher")
        if launcher_project_path:
            from lightning_launcher.__version__ import __version__ as launcher_version

            # todo: check why logging.info is missing in outputs
            print(f"Packaged Lightning Launcher with your application. Version: {launcher_version}")
            _prepare_wheel(launcher_project_path)
            tar_name = _copy_tar(launcher_project_path, root)
            tar_files.append(os.path.join(root, tar_name))

        # building and copying lightning-cloud wheel if installed in editable mode
        lightning_cloud_project_path = get_dist_path_if_editable_install("lightning_cloud")
        if lightning_cloud_project_path:
            from lightning_cloud.__version__ import __version__ as cloud_version

            # todo: check why logging.info is missing in outputs
            print(f"Packaged Lightning Cloud with your application. Version: {cloud_version}")
            _prepare_wheel(lightning_cloud_project_path)
            tar_name = _copy_tar(lightning_cloud_project_path, root)
            tar_files.append(os.path.join(root, tar_name))

    return functools.partial(_cleanup, *tar_files)


def _enable_debugging():
    tar_file = os.path.join(os.getcwd(), f"lightning-{version}.tar.gz")

    if not os.path.exists(tar_file):
        return

    _root_logger.propagate = True
    _logger.propagate = True
    _root_logger.setLevel(logging.DEBUG)
    _root_logger.debug("Setting debugging mode.")


def enable_debugging(func: Callable) -> Callable:
    """This function is used to transform any print into logger.info calls, so it gets tracked in the cloud."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _enable_debugging()
        res = func(*args, **kwargs)
        _logger.setLevel(logging.INFO)
        return res

    return wrapper


def _fetch_latest_version(package_name: str) -> str:
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"{package_name}==1000",
    ]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, close_fds=True)
    if proc.stdout:
        logs = " ".join([line.decode("utf-8") for line in iter(proc.stdout.readline, b"")])
        return logs.split(")\n")[0].split(",")[-1].replace(" ", "")
    return version


def _verify_lightning_version():
    """This function verifies that users are running the latest lightning version for the cloud."""
    # TODO (tchaton) Add support for windows
    if sys.platform == "win32":
        return

    lightning_latest_version = _fetch_latest_version("lightning")

    if Version(lightning_latest_version) > Version(version):
        raise Exception(
            f"You need to use the latest version of Lightning ({lightning_latest_version}) to run in the cloud. "
            "Please, run `pip install -U lightning`"
        )

import concurrent
import os
import sys
from functools import partial
from multiprocessing.pool import ApplyResult
from pathlib import Path
from time import sleep
from typing import Optional, Tuple

import click
import requests
import rich
import urllib3
from lightning_cloud.openapi import IdArtifactsBody
from rich.live import Live
from rich.progress import BarColumn, DownloadColumn, Progress, Task, TextColumn
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.commands.pwd import _pwd
from lightning.app.source_code import FileUploader
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.network import LightningClient

logger = Logger(__name__)


@click.argument("src_path", required=True)
@click.argument("dst_path", required=True)
@click.option("-r", required=False, hidden=True)
@click.option("--recursive", required=False, hidden=True)
def cp(src_path: str, dst_path: str, r: bool = False, recursive: bool = False) -> None:
    """Command to copy files between your local and the Lightning Cloud filesystem's."""

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True) as live:

        pwd = _pwd()

        if pwd == "/" or len(pwd.split("/")) == 1:
            return _error_and_exit("Uploading files at the project level isn't supported yet.")

        client = LightningClient()

        src_path, src_remote = _sanetize_path(src_path, pwd)
        dst_path, dst_remote = _sanetize_path(dst_path, pwd)

        if src_remote and dst_remote:
            return _error_and_exit("Moving files remotely isn't support yet. Please, open a Github issue.")

        if not src_remote and dst_remote:
            _upload_files(live, client, src_path, dst_path, pwd)
        elif src_remote and not dst_remote:
            _download_files(live, client, src_path, dst_path, pwd)
        else:
            return _error_and_exit("Moving files locally isn't support yet. Please, open a Github issue.")


def _upload_files(live, client: LightningClient, local_src: str, remote_dst: str, pwd: str) -> str:
    if not os.path.exists(local_src):
        return _error_and_exit(f"The provided source path {local_src} doesn't exist.")

    project_id, app_id = _get_project_app_ids(pwd)

    local_src = Path(local_src).resolve()
    upload_paths = []

    if os.path.isdir(local_src):
        for root_dir, _, paths in os.walk(local_src):
            for path in paths:
                upload_paths.append(os.path.join(root_dir, path))
    else:
        upload_paths = [local_src]

    upload_urls = []

    for upload_path in upload_paths:
        filename = str(upload_path).replace(str(os.getcwd()), "")[1:]
        response = client.lightningapp_instance_service_upload_lightningapp_instance_artifact(
            project_id=project_id,
            id=app_id,
            body=IdArtifactsBody(filename),
            async_req=True,
        )
        upload_urls.append(response)

    live.stop()

    sleep(1)

    progress = _get_progress_bar()

    total_size = sum([Path(path).stat().st_size for path in upload_paths])
    task_id = progress.add_task("upload", filename="", total=total_size)

    progress.start()

    _upload_partial = partial(_upload, progress=progress, task_id=task_id)

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        results = executor.map(_upload_partial, upload_paths, upload_urls)

    progress.stop()

    # Raise the first exception found
    exception = next((e for e in results if isinstance(e, Exception)), None)
    if exception:
        _error_and_exit("We detected errors in uploading your files.")


def _upload(source_file: str, presigned_url: ApplyResult, progress: Progress, task_id: Task) -> Optional[Exception]:
    source_file = Path(source_file)
    file_uploader = FileUploader(
        presigned_url.get().upload_url,
        source_file,
        total_size=None,
        name=str(source_file),
    )
    file_uploader.progress = progress
    file_uploader.task_id = task_id
    file_uploader.upload()


def _download_files(live, client, remote_src: str, local_dst: str, pwd: str):
    project_id, app_id = _get_project_app_ids(pwd)

    download_paths = []
    download_urls = []
    total_size = []

    response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(project_id, app_id)
    for artifact in response.artifacts:
        path = os.path.join(local_dst, artifact.filename.replace(remote_src, ""))
        path = Path(path).resolve()
        os.makedirs(path.parent, exist_ok=True)
        download_paths.append(Path(path).resolve())
        download_urls.append(artifact.url)
        total_size.append(int(artifact.size_bytes))

    live.stop()

    # Sleep to avoid rich live collision.
    sleep(1)

    progress = progress = _get_progress_bar()

    progress.start()

    task_id = progress.add_task("download", filename=path, total=sum(total_size))

    _download_file_fn = partial(_download_file, progress=progress, task_id=task_id)

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        results = executor.map(_download_file_fn, download_paths, download_urls)

    progress.stop()

    # Raise the first exception found
    exception = next((e for e in results if isinstance(e, Exception)), None)
    if exception:
        _error_and_exit("We detected errors in downloading your files.")


def _download_file(path: str, url: str, progress: Progress, task_id: Task) -> None:
    # Disable warning about making an insecure request
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    request = requests.get(url, stream=True, verify=False)

    chunk_size = 1024

    with open(path, "wb") as fp:
        for chunk in request.iter_content(chunk_size=chunk_size):
            fp.write(chunk)  # type: ignore
            progress.update(task_id, advance=len(chunk))


def _sanetize_path(path: str, pwd: str) -> Tuple[str, bool]:
    is_remote = _is_remote(path)
    if is_remote:
        path = _remove_remote(path)
        if path == ".":
            path = pwd
        else:
            path = os.path.join(pwd, path[1:])
    return path, is_remote


def _is_remote(path: str) -> bool:
    return path.startswith("r:") or path.startswith("remote:")


def _remove_remote(path: str) -> str:
    return path.replace("r:", "").replace("remote:", "")


def _error_and_exit(msg: str) -> str:
    rich.print(f"[red]ERROR[/red]: {msg}")
    sys.exit(0)


def _get_project_app_ids(pwd: str) -> Tuple[str, str]:
    """Convert a root path to a project id and app id."""
    project_name, app_name, *_ = pwd.split("/")[1:]
    client = LightningClient()
    projects = client.projects_service_list_memberships()
    project_id = [project.project_id for project in projects.memberships if project.name == project_name][0]
    client = LightningClient()
    lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id).lightningapps
    lit_apps = [lit_app for lit_app in lit_apps if lit_app.name == app_name]
    assert len(lit_apps) == 1
    lit_app = lit_apps[0]
    return project_id, lit_app.id


def _get_progress_bar():
    return Progress(
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        "[self.progress.percentage]{task.percentage:>3.1f}%",
        DownloadColumn(),
    )

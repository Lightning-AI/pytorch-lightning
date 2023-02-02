import concurrent
import os
import sys
from multiprocessing.pool import ApplyResult
from pathlib import Path
from typing import Optional, Tuple

import click
import requests
import rich
import urllib3
from lightning_cloud.openapi import IdArtifactsBody
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER
from lightning.app.source_code import FileUploader
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient

logger = Logger(__name__)


@click.argument("src_path", required=True)
@click.argument("dst_path", required=True)
@click.option("--project_id", required=False)
@click.option("-r", required=False)
@click.option("--recursive", required=False)
def cp(
    src_path: str, dst_path: str, project_id: Optional[str] = None, r: bool = False, recursive: bool = False
) -> None:
    cd_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "cd.txt")
    pwd = "/"

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True) as live:

        if os.path.exists(cd_file):
            with open(cd_file) as f:
                lines = f.readlines()
                pwd = lines[0].replace("\n", "")

        if pwd == "/":
            print("ERROR: Uploading files at the project level isn't supported yet.")
            sys.exit(0)

        app_name = pwd.split("/")[1]

        client = LightningClient()

        if not project_id:
            project_id = _get_project(client, verbose=False).project_id

        lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id).lightningapps

        lit_apps = [lit_app for lit_app in lit_apps if lit_app.name == app_name]

        if len(lit_apps) == 0:
            print(f"ERROR: We didn't find an app with the name {app_name}. HINT: Did you try using `--project_id`.")
            sys.exit(0)

        lit_app = lit_apps[0]

        src_path, src_remote = _sanetize_path(src_path, pwd)
        dst_path, dst_remote = _sanetize_path(dst_path, pwd)

        if src_remote and dst_remote:
            print("ERROR: Moving files remotely isn't support yet. Please, open a Github issue.")
            sys.exit(0)

        if not src_remote and dst_remote:
            _upload_files(live, client, src_path, dst_path, project_id, lit_app.id)
        elif src_remote and not dst_remote:
            _download_files(live, client, src_path, dst_path, project_id, lit_app.id)
        else:
            print("ERROR: Moving files locally isn't support yet. Please, open a Github issue.")
            sys.exit(0)


def _upload_files(live, client: LightningClient, local_src: str, remote_dst: str, project_id: str, app_id: str) -> str:
    if not os.path.exists(local_src):
        print(f"ERROR: The provided source path {local_src} doesn't exist.")
        sys.exit(0)

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

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        results = executor.map(_upload, upload_paths, upload_urls)

    # Raise the first exception found
    exception = next((e for e in results if isinstance(e, Exception)), None)
    if exception:
        raise exception


def _upload(source_file: str, presigned_url: ApplyResult) -> Optional[Exception]:
    source_file = Path(source_file)
    file_uploader = FileUploader(
        presigned_url.get().upload_url,
        source_file,
        name=str(source_file),
        total_size=source_file.stat().st_size,
    )
    file_uploader.upload()


def _download_files(live, client, remote_src: str, local_dst: str, project_id: str, app_id: str):
    download_paths = []
    download_urls = []

    response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(project_id, app_id)
    for artifact in response.artifacts:
        path = os.path.join(local_dst, artifact.filename.replace(remote_src, ""))
        path = Path(path).resolve()
        os.makedirs(path.parent, exist_ok=True)
        download_paths.append(Path(path).resolve())
        download_urls.append(artifact.url)

    live.stop()

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        results = executor.map(_download_file, download_paths, download_urls)

    # Raise the first exception found
    exception = next((e for e in results if isinstance(e, Exception)), None)
    if exception:
        raise exception


def _download_file(path: str, url: str) -> None:
    r = requests.get(url, stream=True, verify=False)
    file_size = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        "[self.progress.percentage]{task.percentage:>3.1f}%",
    )

    task_id = progress.add_task("upload", filename=path, total=file_size)
    # TODO: Resolve the live error
    try:
        progress.start()
    except rich.errors.LiveError:
        pass

    # Disable warning about making an insecure request
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    task_id = progress.add_task("upload", filename=path, total=file_size)
    chunk_size = 1024

    if not os.path.exists(path):
        with open(path, "wb") as fp:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fp.write(chunk)  # type: ignore
                progress.update(task_id, advance=chunk_size)

    progress.stop()


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

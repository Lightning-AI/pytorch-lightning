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

import concurrent
import contextlib
import os
import sys
from functools import partial
from multiprocessing.pool import ApplyResult
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional, Tuple, Union

import click
import requests
import urllib3
from lightning_cloud.openapi import (
    Externalv1Cluster,
    Externalv1LightningappInstance,
    ProjectIdStorageBody,
    V1CloudSpace,
)
from rich.live import Live
from rich.progress import BarColumn, DownloadColumn, Progress, TaskID, TextColumn
from rich.spinner import Spinner
from rich.text import Text

from lightning.app.cli.commands.ls import _collect_artifacts, _get_prefix
from lightning.app.cli.commands.pwd import _pwd
from lightning.app.source_code import FileUploader
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.auth import _AuthTokenGetter
from lightning.app.utilities.cli_helpers import _error_and_exit
from lightning.app.utilities.network import LightningClient

logger = Logger(__name__)


@click.argument("src_path", required=True)
@click.argument("dst_path", required=True)
@click.option("-r", required=False, hidden=True)
@click.option("--recursive", required=False, hidden=True)
@click.option("--zip", required=False, is_flag=True, default=False)
def cp(src_path: str, dst_path: str, r: bool = False, recursive: bool = False, zip: bool = False) -> None:
    """Copy files between your local filesystem and the Lightning Cloud filesystem."""
    if sys.platform == "win32":
        print("`cp` isn't supported on windows. Open an issue on Github.")
        sys.exit(0)

    with Live(Spinner("point", text=Text("pending...", style="white")), transient=True) as live:
        pwd = _pwd()

        client = LightningClient(retry=False)

        src_path, src_remote = _sanitize_path(src_path, pwd)
        dst_path, dst_remote = _sanitize_path(dst_path, pwd)

        if src_remote and dst_remote:
            return _error_and_exit("Moving files remotely isn't supported yet. Please, open a Github issue.")

        if not src_remote and dst_remote:
            if dst_path == "/" or len(dst_path.split("/")) == 1:
                return _error_and_exit("Uploading files at the project level isn't allowed yet.")
            if zip:
                return _error_and_exit("Zipping uploads isn't supported yet. Please, open a Github issue.")
            _upload_files(live, client, src_path, dst_path, pwd)
            return None
        if src_remote and not dst_remote:
            if zip:
                return _zip_files(live, src_path, dst_path)
            _download_files(live, client, src_path, dst_path, pwd)
            return None

        return _error_and_exit("Moving files locally isn't supported yet. Please, open a Github issue.")


def _upload_files(live, client: LightningClient, local_src: str, remote_dst: str, pwd: str) -> str:
    remote_splits = [split for split in remote_dst.split("/") if split != ""]
    remote_dst = os.path.join(*remote_splits)

    if not os.path.exists(local_src):
        return _error_and_exit(f"The provided source path {local_src} doesn't exist.")

    lit_resource = None

    if len(remote_splits) > 1:
        project_id, lit_resource = _get_project_id_and_resource(pwd)
    else:
        project_id = _get_project_id_from_name(remote_dst)

    if len(remote_splits) > 2:
        remote_dst = os.path.join(*remote_splits[2:])

    local_src = Path(local_src).resolve()
    upload_paths = []

    if os.path.isdir(local_src):
        for root_dir, _, paths in os.walk(local_src):
            for path in paths:
                upload_paths.append(os.path.join(root_dir, path))
    else:
        upload_paths = [local_src]

    _upload_urls = []

    clusters = client.projects_service_list_project_cluster_bindings(project_id)

    live.stop()

    for upload_path in upload_paths:
        for cluster in clusters.clusters:
            filename = str(upload_path).replace(str(os.getcwd()), "")[1:]
            filename = _get_prefix(os.path.join(remote_dst, filename), lit_resource) if lit_resource else "/" + filename

            response = client.lightningapp_instance_service_upload_project_artifact(
                project_id=project_id,
                body=ProjectIdStorageBody(cluster_id=cluster.cluster_id, filename=filename),
                async_req=True,
            )
            _upload_urls.append(response)

    upload_urls = []
    for upload_url in _upload_urls:
        upload_urls.extend(upload_url.get().urls)

    live.stop()

    if not upload_paths:
        print("There were no files to upload.")
        return None

    progress = _get_progress_bar()

    total_size = sum([Path(path).stat().st_size for path in upload_paths]) // max(len(clusters.clusters), 1)
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
        return None
    return None


def _upload(source_file: str, presigned_url: ApplyResult, progress: Progress, task_id: TaskID) -> Optional[Exception]:
    source_file = Path(source_file)
    file_uploader = FileUploader(
        presigned_url,
        source_file,
        total_size=None,
        name=str(source_file),
    )
    file_uploader.progress = progress
    file_uploader.task_id = task_id
    file_uploader.upload()


def _zip_files(live: Live, remote_src: str, local_dst: str) -> None:
    if len(remote_src.split("/")) < 3:
        return _error_and_exit(
            dedent(
                f"""
                The source path must be at least two levels deep (e.g. r:/my-project/my-lit-resource).

                The path provided was: r:{remote_src}
                """
            )
        )

    if os.path.isdir(local_dst):
        local_dst = os.path.join(local_dst, os.path.basename(remote_src) + ".zip")

    project_id, lit_resource = _get_project_id_and_resource(remote_src)

    # /my-project/my-lit-resource/artfact-path -> cloudspace/my-lit-resource-id/artifact-path
    artifact = "/".join(remote_src.split("/")[3:])
    prefix = _get_prefix(artifact, lit_resource)

    token = _AuthTokenGetter(LightningClient().api_client)._get_api_token()
    endpoint = f"/v1/projects/{project_id}/artifacts/download?prefix={prefix}&token={token}"

    cluster = _cluster_from_lit_resource(lit_resource)
    url = _storage_host(cluster) + endpoint

    live.stop()
    progress = _get_progress_bar(transient=True)
    progress.start()
    task_id = progress.add_task("download zip", total=None)

    _download_file(local_dst, url, progress, task_id)
    progress.stop()

    click.echo(f"Downloaded to {local_dst}")
    return None


def _download_files(live, client, remote_src: str, local_dst: str, pwd: str):
    project_id, lit_resource = _get_project_id_and_resource(pwd)

    download_paths = []
    download_urls = []
    total_size = []

    prefix = _get_prefix("/".join(pwd.split("/")[3:]), lit_resource) + "/"

    for artifact in _collect_artifacts(client, project_id, prefix, include_download_url=True):
        path = os.path.join(local_dst, artifact.filename.replace(remote_src, ""))
        path = Path(path).resolve()
        os.makedirs(path.parent, exist_ok=True)
        download_paths.append(path)
        download_urls.append(artifact.url)
        total_size.append(int(artifact.size_bytes))

    live.stop()

    if not download_paths:
        print("There were no files to download.")
        return

    progress = progress = _get_progress_bar()

    progress.start()

    task_id = progress.add_task("download", filename="", total=sum(total_size))

    _download_file_fn = partial(_download_file, progress=progress, task_id=task_id)

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        results = executor.map(_download_file_fn, download_paths, download_urls)

    progress.stop()

    # Raise the first exception found
    exception = next((e for e in results if isinstance(e, Exception)), None)
    if exception:
        _error_and_exit("There was an error downloading your files.")


def _download_file(path: str, url: str, progress: Progress, task_id: TaskID) -> None:
    # Disable warning about making an insecure request
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    with contextlib.suppress(ConnectionError):
        request = requests.get(url, stream=True, verify=False)  # noqa: S501

        chunk_size = 1024

        with open(path, "wb") as fp:
            for chunk in request.iter_content(chunk_size=chunk_size):
                fp.write(chunk)  # type: ignore
                progress.update(task_id, advance=len(chunk))


def _sanitize_path(path: str, pwd: str) -> Tuple[str, bool]:
    is_remote = _is_remote(path)
    if is_remote:
        path = _remove_remote(path)
        path = pwd if path == "." else os.path.join(pwd, path)
    return path, is_remote


def _is_remote(path: str) -> bool:
    return path.startswith("r:") or path.startswith("remote:")


def _remove_remote(path: str) -> str:
    return path.replace("r:", "").replace("remote:", "")


def _get_project_id_and_resource(pwd: str) -> Tuple[str, Union[Externalv1LightningappInstance, V1CloudSpace]]:
    """Convert a root path to a project id and app id."""
    # TODO: Handle project level
    project_name, resource_name, *_ = pwd.split("/")[1:3]

    # 1. Collect the projects of the user
    client = LightningClient()
    projects = client.projects_service_list_memberships()
    project_id = [project.project_id for project in projects.memberships if project.name == project_name][0]

    # 2. Collect resources
    lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id).lightningapps

    lit_cloud_spaces = client.cloud_space_service_list_cloud_spaces(project_id=project_id).cloudspaces

    lit_ressources = [lit_resource for lit_resource in lit_cloud_spaces if lit_resource.name == resource_name]

    if len(lit_ressources) == 0:
        lit_ressources = [lit_resource for lit_resource in lit_apps if lit_resource.name == resource_name]

        if len(lit_ressources) == 0:
            print(f"ERROR: There isn't any Lightning Ressource matching the name {resource_name}.")
            sys.exit(0)

    return project_id, lit_ressources[0]


def _get_project_id_from_name(project_name: str) -> str:
    # 1. Collect the projects of the user
    client = LightningClient()
    projects = client.projects_service_list_memberships()
    return [project.project_id for project in projects.memberships if project.name == project_name][0]


def _get_progress_bar(**kwargs: Any) -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        "[self.progress.percentage]{task.percentage:>3.1f}%",
        DownloadColumn(),
        **kwargs,
    )


def _storage_host(cluster: Externalv1Cluster) -> str:
    dev_host = os.environ.get("LIGHTNING_STORAGE_HOST")
    if dev_host:
        return dev_host
    return f"https://storage.{cluster.spec.driver.kubernetes.root_domain_name}"


def _cluster_from_lit_resource(lit_resource: Union[Externalv1LightningappInstance, V1CloudSpace]) -> Externalv1Cluster:
    client = LightningClient()
    if isinstance(lit_resource, Externalv1LightningappInstance):
        return client.cluster_service_get_cluster(lit_resource.spec.cluster_id)

    clusters = client.cluster_service_list_clusters()
    for cluster in clusters.clusters:
        if cluster.id == clusters.default_cluster:
            return cluster
    return None

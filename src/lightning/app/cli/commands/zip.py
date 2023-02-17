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
import sys
from typing import Union
from textwrap import dedent
from pathlib import Path

import click
from lightning_cloud.openapi import (
    Externalv1Cluster,
    Externalv1LightningappInstance,
    V1CloudSpace,
    V1GetClusterResponse,
)

from lightning.app.cli.commands.cp import (
    _download_file,
    _get_progress_bar,
    _get_project_id_and_resource,
    _sanitize_path,
    _is_remote,
)
from lightning.app.cli.commands.ls import _get_prefix
from lightning.app.cli.commands.pwd import _pwd
from lightning.app.utilities.auth import _AuthTokenGetter
from lightning.app.utilities.cli_helpers import _error_and_exit
from lightning.app.utilities.network import LightningClient


@click.argument("src_path", required=True)
@click.argument("dst_path", required=False, default=".")
def zip(src_path: str, dst_path: str = ".") -> None:
    """Download content from the Lightning Filesystem as a zip file."""

    if sys.platform == "win32":
        print("`cp` isn't supported on windows. Open an issue on Github.")
        sys.exit(0)

    pwd = _pwd()

    if not _is_remote(src_path):
        src_path = "r:" + src_path

    src_path, _ = _sanitize_path(src_path, pwd)
    dst_path, dst_is_remote = _sanitize_path(dst_path, pwd)

    if dst_is_remote:
        return _error_and_exit(
            dedent(
                """
                The destination path must be a local path (i.e. not prefixed with [red]r:[/red] or [red]remote:[/red]).

                The zip command only supports downloading from the Lightning Filesystem.
                For other use cases, please open a Github issue.
                """
            )
        )

    if os.path.isdir(dst_path):
        dst_path = os.path.join(dst_path, os.path.basename(src_path) + ".zip")

    if len(src_path.split("/")) < 3:
        return _error_and_exit(
            dedent(
                f"""
                The source path must be at least two levels deep (e.g. r:/my-project/my-lit-resource).

                The path provided was: r:{src_path}
                """
            )
        )

    return _zip_files(src_path, dst_path)


def _zip_files(remote_src: str, local_dst: str) -> None:
    project_id, lit_resource = _get_project_id_and_resource(remote_src)

    # /my-project/my-lit-resource/artfact-path -> cloudspace/my-lit-resource-id/artifact-path
    artifact = "/".join(remote_src.split("/")[3:])
    prefix = _get_prefix(artifact, lit_resource)

    token = _AuthTokenGetter(LightningClient().api_client)._get_api_token()
    endpoint = f"/v1/projects/{project_id}/artifacts/download?prefix={prefix}&token={token}"

    cluster = _cluster_from_lit_resource(lit_resource)
    url = _storage_host(cluster) + endpoint

    progress = _get_progress_bar(transient=True)
    progress.start()
    task_id = progress.add_task("download zip", total=None)

    _download_file(local_dst, url, progress, task_id)
    progress.stop()

    click.echo(f"Downloaded to {local_dst}")


def _storage_host(cluster: Union[V1GetClusterResponse, Externalv1Cluster]) -> str:
    if os.environ.get("LIGHTNING_STORAGE_HOST"):
        return os.environ.get("LIGHTNING_STORAGE_HOST")
    return f"https://storage.{cluster.spec.driver.kubernetes.root_domain_name}"


def _cluster_from_lit_resource(
    lit_resource: Union[Externalv1LightningappInstance, V1CloudSpace]
) -> Union[V1GetClusterResponse, Externalv1Cluster]:
    client = LightningClient()
    if isinstance(lit_resource, Externalv1LightningappInstance):
        return client.cluster_service_get_cluster(lit_resource.spec.cluster_id)

    clusters = client.cluster_service_list_clusters()
    for cluster in clusters.clusters:
        if cluster.id == clusters.default_cluster:
            return cluster

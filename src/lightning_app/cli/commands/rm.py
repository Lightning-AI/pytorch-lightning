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

import click
import lightning_cloud
import rich

from lightning_app.cli.commands.ls import _add_colors, _get_prefix
from lightning_app.cli.commands.pwd import _pwd
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.cli_helpers import _error_and_exit
from lightning_app.utilities.network import LightningClient

logger = Logger(__name__)


@click.argument("rm_path", required=True)
@click.option("-r", required=False, hidden=True)
@click.option("--recursive", required=False, hidden=True)
def rm(rm_path: str, r: bool = False, recursive: bool = False) -> None:
    """Delete files on the Lightning Cloud filesystem."""

    root = _pwd()

    if rm_path in (".", ".."):
        return _error_and_exit('rm "." and ".." may not be removed')

    if ".." in rm_path:
        return _error_and_exit('rm ".." or higher may not be removed')

    root = os.path.join(root, rm_path)
    splits = [split for split in root.split("/") if split != ""]

    if root == "/" or len(splits) == 1:
        return _error_and_exit("rm at the project level isn't supported")

    client = LightningClient(retry=False)
    projects = client.projects_service_list_memberships()

    project = [project for project in projects.memberships if project.name == splits[0]]

    # This happens if the user changes cluster and the project doesn't exist.
    if len(project) == 0:
        return _error_and_exit(
            f"There isn't any Lightning Project matching the name {splits[0]}." " HINT: Use `lightning cd`."
        )

    project_id = project[0].project_id

    # Parallelise calls
    lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id, async_req=True)
    lit_cloud_spaces = client.cloud_space_service_list_cloud_spaces(project_id=project_id, async_req=True)

    lit_apps = lit_apps.get().lightningapps
    lit_cloud_spaces = lit_cloud_spaces.get().cloudspaces

    lit_ressources = [lit_resource for lit_resource in lit_cloud_spaces if lit_resource.name == splits[1]]

    if len(lit_ressources) == 0:

        lit_ressources = [lit_resource for lit_resource in lit_apps if lit_resource.name == splits[1]]

        if len(lit_ressources) == 0:
            _error_and_exit(f"There isn't any Lightning Ressource matching the name {splits[1]}.")

    lit_resource = lit_ressources[0]

    prefix = "/".join(splits[2:])
    prefix = _get_prefix(prefix, lit_resource)

    clusters = client.projects_service_list_project_cluster_bindings(project_id)
    succeeded = False

    for cluster in clusters.clusters:
        try:
            client.lightningapp_instance_service_delete_project_artifact(
                project_id=project_id,
                cluster_id=cluster.cluster_id,
                filename=prefix,
            )
            succeeded = True
            break
        except lightning_cloud.openapi.rest.ApiException:
            pass

    prefix = os.path.join(*splits)

    if succeeded:
        rich.print(_add_colors(f"Successfuly deleted `{prefix}`.", color="green"))
    else:
        return _error_and_exit(f"No file or folder named `{prefix}` was found.")

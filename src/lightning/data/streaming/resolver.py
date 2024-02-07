import datetime
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Optional, Union
from urllib import parse

from lightning_cloud.openapi import V1CloudSpace
from lightning_cloud.rest_client import LightningClient

# To avoid adding lightning_utilities as a dependency for now.
try:
    import boto3
    import botocore

    _BOTO3_AVAILABLE = True
except Exception:
    _BOTO3_AVAILABLE = False


try:
    from lightning_sdk import Machine, Studio

    _LIGHTNING_SDK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):

    class Machine:  # type: ignore
        pass

    _LIGHTNING_SDK_AVAILABLE = False


@dataclass
class Dir:
    """Holds a directory path and possibly its associated remote URL."""

    path: Optional[str] = None
    url: Optional[str] = None


def _resolve_dir(dir_path: Optional[Union[str, Dir]]) -> Dir:
    if isinstance(dir_path, Dir):
        return Dir(path=str(dir_path.path) if dir_path.path else None, url=str(dir_path.url) if dir_path.url else None)

    if dir_path is None:
        return Dir()

    if not isinstance(dir_path, str):
        raise ValueError(f"`dir_path` must be a `Dir` or a string, got: {dir_path}")

    assert isinstance(dir_path, str)

    if dir_path.startswith("s3://"):
        return Dir(path=None, url=dir_path)

    dir_path = _resolve_time_template(dir_path)

    dir_path_absolute = str(Path(dir_path).absolute().resolve())

    if dir_path_absolute.startswith("/teamspace/studios/this_studio"):
        return Dir(path=dir_path_absolute, url=None)

    if dir_path_absolute.startswith("/.project"):
        dir_path_absolute = dir_path

    if dir_path_absolute.startswith("/teamspace/studios") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_studio(dir_path_absolute, dir_path_absolute.split("/")[3], None)

    if dir_path_absolute.startswith("/teamspace/s3_connections") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_s3_connections(dir_path_absolute)

    if dir_path_absolute.startswith("/teamspace/datasets") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_datasets(dir_path_absolute)

    return Dir(path=dir_path_absolute, url=None)


def _match_studio(target_id: Optional[str], target_name: Optional[str], cloudspace: V1CloudSpace) -> bool:
    if cloudspace.name is not None and target_name is not None and cloudspace.name.lower() == target_name.lower():
        return True

    if target_id is not None and cloudspace.id == target_id:
        return True

    if (
        cloudspace.display_name is not None
        and target_name is not None
        and cloudspace.display_name.lower() == target_name.lower()
    ):
        return True

    return False


def _resolve_studio(dir_path: str, target_name: Optional[str], target_id: Optional[str]) -> Dir:
    client = LightningClient(max_tries=2)

    # Get the ids from env variables
    cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)

    if cluster_id is None:
        raise RuntimeError("The `cluster_id` couldn't be found from the environement variables.")

    if project_id is None:
        raise RuntimeError("The `project_id` couldn't be found from the environement variables.")

    clusters = client.cluster_service_list_project_clusters(project_id).clusters

    cloudspaces = client.cloud_space_service_list_cloud_spaces(project_id=project_id, cluster_id=cluster_id).cloudspaces
    target_cloud_space = [cloudspace for cloudspace in cloudspaces if _match_studio(target_id, target_name, cloudspace)]

    if not target_cloud_space:
        raise ValueError(f"We didn't find any matching Studio for the provided name `{target_name}`.")

    target_cluster = [cluster for cluster in clusters if cluster.id == target_cloud_space[0].cluster_id]

    if not target_cluster:
        raise ValueError(
            f"We didn't find a matching cluster associated with the id {target_cloud_space[0].cluster_id}."
        )

    bucket_name = target_cluster[0].spec.aws_v1.bucket_name

    return Dir(
        path=dir_path,
        url=os.path.join(
            f"s3://{bucket_name}/projects/{project_id}/cloudspaces/{target_cloud_space[0].id}/code/content",
            *dir_path.split("/")[4:],
        ),
    )


def _resolve_s3_connections(dir_path: str) -> Dir:
    client = LightningClient(max_tries=2)

    # Get the ids from env variables
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    if project_id is None:
        raise RuntimeError("The `project_id` couldn't be found from the environement variables.")

    target_name = dir_path.split("/")[3]

    data_connections = client.data_connection_service_list_data_connections(project_id).data_connections

    data_connection = [dc for dc in data_connections if dc.name == target_name]

    if not data_connection:
        raise ValueError(f"We didn't find any matching data connection with the provided name `{target_name}`.")

    return Dir(path=dir_path, url=os.path.join(data_connection[0].aws.source, *dir_path.split("/")[4:]))


def _resolve_datasets(dir_path: str) -> Dir:
    client = LightningClient(max_tries=2)

    # Get the ids from env variables
    cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    cloud_space_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID", None)

    if cluster_id is None:
        raise RuntimeError("The `cluster_id` couldn't be found from the environement variables.")

    if project_id is None:
        raise RuntimeError("The `project_id` couldn't be found from the environement variables.")

    if cloud_space_id is None:
        raise RuntimeError("The `cloud_space_id` couldn't be found from the environement variables.")

    clusters = client.cluster_service_list_project_clusters(project_id).clusters

    target_cloud_space = [
        cloudspace
        for cloudspace in client.cloud_space_service_list_cloud_spaces(
            project_id=project_id, cluster_id=cluster_id
        ).cloudspaces
        if cloudspace.id == cloud_space_id
    ]

    if not target_cloud_space:
        raise ValueError(f"We didn't find any matching Studio for the provided id `{cloud_space_id}`.")

    target_cluster = [cluster for cluster in clusters if cluster.id == target_cloud_space[0].cluster_id]

    if not target_cluster:
        raise ValueError(
            f"We didn't find a matching cluster associated with the id {target_cloud_space[0].cluster_id}."
        )

    return Dir(
        path=dir_path,
        url=os.path.join(
            f"s3://{target_cluster[0].spec.aws_v1.bucket_name}/projects/{project_id}/datasets/",
            *dir_path.split("/")[3:],
        ),
    )


def _assert_dir_is_empty(output_dir: Dir, append: bool = False, overwrite: bool = False) -> None:
    if not isinstance(output_dir, Dir):
        raise ValueError("The provided output_dir isn't a Dir Object.")

    if output_dir.url is None:
        return

    obj = parse.urlparse(output_dir.url)

    if obj.scheme != "s3":
        raise ValueError(f"The provided folder should start with s3://. Found {output_dir.path}.")

    s3 = boto3.client("s3")

    objects = s3.list_objects_v2(
        Bucket=obj.netloc,
        Delimiter="/",
        Prefix=obj.path.lstrip("/").rstrip("/") + "/",
    )

    # We aren't alloweing to add more data
    # TODO: Add support for `append` and `overwrite`.
    if objects["KeyCount"] > 0:
        raise RuntimeError(
            f"The provided output_dir `{output_dir.path}` already contains data and datasets are meant to be immutable."
            " HINT: Did you consider changing the `output_dir` with your own versioning as a suffix?"
        )


def _assert_dir_has_index_file(output_dir: Dir) -> None:
    if not isinstance(output_dir, Dir):
        raise ValueError("The provided output_dir isn't a Dir Object.")

    if output_dir.url is None:
        return

    obj = parse.urlparse(output_dir.url)

    if obj.scheme != "s3":
        raise ValueError(f"The provided folder should start with s3://. Found {output_dir.path}.")

    s3 = boto3.client("s3")

    prefix = obj.path.lstrip("/").rstrip("/") + "/"

    objects = s3.list_objects_v2(
        Bucket=obj.netloc,
        Delimiter="/",
        Prefix=prefix,
    )

    # No files are found in this folder
    if objects["KeyCount"] == 0:
        return

    # Check the index file exists
    try:
        s3.head_object(Bucket=obj.netloc, Key=os.path.join(prefix, "index.json"))
        has_index_file = True
    except botocore.exceptions.ClientError:
        has_index_file = False

    if has_index_file:
        raise RuntimeError(
            f"The provided output_dir `{output_dir.path}` already contains an optimized immutable datasets."
            " HINT: Did you consider changing the `output_dir` with your own versioning as a suffix?"
        )

    bucket_name = obj.netloc
    s3 = boto3.resource("s3")
    for obj in s3.Bucket(bucket_name).objects.filter(Prefix=prefix):
        s3.Object(bucket_name, obj.key).delete()


def _get_lightning_cloud_url() -> str:
    # detect local development
    if os.getenv("VSCODE_PROXY_URI", "").startswith("http://localhost:9800"):
        return "http://localhost:9800"
    # DO NOT CHANGE!
    return os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")


def _resolve_time_template(path: str) -> str:
    match = re.search("^.*{%.*}$", path)
    if match is None:
        return path

    pattern = path.split("{")[1].split("}")[0]

    return path.replace("{" + pattern + "}", datetime.datetime.now().strftime(pattern))


def _execute(
    name: str,
    num_nodes: int,
    machine: Optional[Machine] = None,
    command: Optional[str] = None,
) -> None:
    """Remotely execute the current operator."""

    if not _LIGHTNING_SDK_AVAILABLE:
        raise ModuleNotFoundError("The `lightning_sdk` is required.")

    lightning_skip_install = os.getenv("LIGHTNING_SKIP_INSTALL", "")
    if lightning_skip_install:
        lightning_skip_install = f" LIGHTNING_SKIP_INSTALL={lightning_skip_install} "

    lightning_branch = os.getenv("LIGHTNING_BRANCH", "")
    if lightning_branch:
        lightning_branch = f" LIGHTNING_BRANCH={lightning_branch} "

    studio = Studio()
    job = studio._studio_api.create_data_prep_machine_job(
        command or f"cd {os.getcwd()} &&{lightning_skip_install}{lightning_branch} python {' '.join(sys.argv)}",
        name=name,
        num_instances=num_nodes,
        studio_id=studio._studio.id,
        teamspace_id=studio._teamspace.id,
        cluster_id=studio._studio.cluster_id,
        machine=machine or studio._studio_api.get_machine(studio._studio.id, studio._teamspace.id),
    )

    has_printed = False

    while True:
        curr_job = studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance(
            project_id=studio._teamspace.id, id=job.id
        )
        if not has_printed:
            cloud_url = os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai").replace(":443", "")
            job_url = f"{cloud_url}/{studio.owner}/{studio._teamspace.name}"
            job_url += f"/studios/{studio.name}/app?app_id=data-prep&job_name={curr_job.name}"
            print(f"Find your job at {job_url}")
            has_printed = True

        if curr_job.status.phase == "LIGHTNINGAPP_INSTANCE_STATE_FAILED":
            raise RuntimeError(f"job {curr_job.name} failed!")

        if curr_job.status.phase in ["LIGHTNINGAPP_INSTANCE_STATE_STOPPED", "LIGHTNINGAPP_INSTANCE_STATE_COMPLETED"]:
            break

        sleep(1)

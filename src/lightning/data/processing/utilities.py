import io
import os
import urllib
from contextlib import contextmanager
from subprocess import DEVNULL, Popen
from typing import Any, Callable, List, Optional, Tuple, Union

from lightning.data.constants import _IS_IN_STUDIO, _LIGHTNING_CLOUD_LATEST

if _LIGHTNING_CLOUD_LATEST:
    from lightning_cloud.openapi import (
        ProjectIdDatasetsBody,
        V1DatasetType,
    )
    from lightning_cloud.openapi.rest import ApiException
    from lightning_cloud.rest_client import LightningClient


def _create_dataset(
    input_dir: Optional[str],
    storage_dir: str,
    dataset_type: V1DatasetType,
    empty: Optional[bool] = None,
    size: Optional[int] = None,
    num_bytes: Optional[str] = None,
    data_format: Optional[Union[str, Tuple[str]]] = None,
    compression: Optional[str] = None,
    num_chunks: Optional[int] = None,
    num_bytes_per_chunk: Optional[List[int]] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> None:
    """Create a dataset with metadata information about its source and destination."""
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
    user_id = os.getenv("LIGHTNING_USER_ID", None)
    cloud_space_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID", None)
    lightning_app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)

    if project_id is None:
        return

    if not storage_dir:
        raise ValueError("The storage_dir should be defined.")

    client = LightningClient(retry=False)

    try:
        client.dataset_service_create_dataset(
            body=ProjectIdDatasetsBody(
                cloud_space_id=cloud_space_id if lightning_app_id is None else None,
                cluster_id=cluster_id,
                creator_id=user_id,
                empty=empty,
                input_dir=input_dir,
                lightning_app_id=lightning_app_id,
                name=name,
                size=size,
                num_bytes=num_bytes,
                data_format=str(data_format) if data_format else data_format,
                compression=compression,
                num_chunks=num_chunks,
                num_bytes_per_chunk=num_bytes_per_chunk,
                storage_dir=storage_dir,
                type=dataset_type,
                version=version,
            ),
            project_id=project_id,
        )
    except ApiException as ex:
        if "already exists" in str(ex.body):
            pass
        else:
            raise ex


def get_worker_rank() -> Optional[str]:
    return os.getenv("DATA_OPTIMIZER_GLOBAL_RANK")


def catch(func: Callable) -> Callable:
    def _wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, Optional[Exception]]:
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            return None, e

    return _wrapper


# Credit to the https://github.com/rom1504/img2dataset Github repo
# The code was taken from there. It has a MIT License.


def make_request(
    url: str,
    timeout: int = 10,
    user_agent_token: str = "pytorch-lightning",
) -> io.BytesIO:
    """Download an image with urllib."""
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/Lightning-AI/pytorch-lightning)"

    with urllib.request.urlopen(  # noqa: S310
        urllib.request.Request(url, data=None, headers={"User-Agent": user_agent_string}), timeout=timeout
    ) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream


@contextmanager
def optimize_dns_context(enable: bool) -> Any:
    optimize_dns(enable)
    try:
        yield
        optimize_dns(False)  # always disable the optimize DNS
    except Exception as e:
        optimize_dns(False)  # always disable the optimize DNS
        raise e


def optimize_dns(enable: bool) -> None:
    if not _IS_IN_STUDIO:
        return

    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    if (enable and any("127.0.0.53" in line for line in lines)) or (
        not enable and any("127.0.0.1" in line for line in lines)
    ):
        cmd = (
            f"sudo /home/zeus/miniconda3/envs/cloudspace/bin/python"
            f" -c 'from lightning.data.processing.utilities import _optimize_dns; _optimize_dns({enable})'"
        )
        Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()  # E501


def _optimize_dns(enable: bool) -> None:
    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    write_lines = []
    for line in lines:
        if "nameserver 127" in line:
            if enable:
                write_lines.append("nameserver 127.0.0.1\n")
            else:
                write_lines.append("nameserver 127.0.0.53\n")
        else:
            write_lines.append(line)

    with open("/etc/resolv.conf", "w") as f:
        for line in write_lines:
            f.write(line)

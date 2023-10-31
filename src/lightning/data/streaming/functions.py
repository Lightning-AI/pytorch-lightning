# Copyright The Lightning AI team.
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
from datetime import datetime
from time import sleep
from typing import Any, Callable, Generator, List, Optional, Sequence, Union

from lightning.app.core.constants import get_lightning_cloud_url
from lightning.data.streaming.constants import _LIGHTNING_CLOUD_GREATER_EQUAL_0_5_42, _LIGHTNING_SDK_AVAILABLE
from lightning.data.streaming.data_processor import DataChunkRecipe, DataProcessor, DataTransformRecipe, PrettyDirectory

if _LIGHTNING_SDK_AVAILABLE:
    from lightning_sdk import Studio


if _LIGHTNING_CLOUD_GREATER_EQUAL_0_5_42:
    from lightning_cloud.resolver import _LightningSrcResolver


class LambdaDataTransformRecipe(DataTransformRecipe):
    def __init__(self, fn: Callable[[str, Any], None], inputs: List[Any]):
        super().__init__()
        self._fn = fn
        self._inputs = inputs

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, output_dir: str, item_metadata: Any) -> None:  # type: ignore
        self._fn(output_dir, item_metadata)


class LambdaDataChunkRecipe(DataChunkRecipe):
    def __init__(self, fn: Callable[[str, Any], None], inputs: List[Any]):
        super().__init__()
        self._fn = fn
        self._inputs = inputs

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, item_metadata: Any) -> Any:  # type: ignore
        if isinstance(self._fn, Generator):
            yield from self._fn(item_metadata)
        else:
            return self._fn(item_metadata)


def map(
    fn: Callable[[str, Any], None],
    inputs: Sequence[Any],
    output_dir: str,
    num_workers: Optional[int] = None,
    fast_dev_run: Union[bool, int] = False,
    num_nodes: Optional[int] = None,
    machine: Optional[str] = None,
) -> None:
    """This function map a callbable over a collection of files possibly in a distributed way.

    Arguments:
        fn: A function to be executed over each input element
        inputs: A sequence of input to be processed by the `fn` function.
            Each input should contain at least a valid filepath.
        output_dir: The folder where the processed data should be stored.
        num_workers: The number of workers to use during processing
        fast_dev_run: Whether to use process only a sub part of the inputs
        num_nodes: When doing remote execution, the number of nodes to use.
        machine: When doing remote execution, the machine to use.

    """

    if num_nodes is None or machine is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        remote_output_dir = _LightningSrcResolver()(output_dir)

        if remote_output_dir is None or "cloudspaces" in remote_output_dir:
            raise ValueError(
                f"The provided `output_dir` isn't valid. Found {output_dir}."
                " HINT: You can either use `/teamspace/s3_connections/...` or `/teamspace/datasets/...`."
            )

        data_processor = DataProcessor(
            name=None,
            num_workers=num_workers or os.cpu_count(),
            remote_output_dir=PrettyDirectory(output_dir, remote_output_dir),
            fast_dev_run=fast_dev_run,
            version=None,
        )
        data_processor.run(LambdaDataTransformRecipe(fn, inputs() if callable(inputs) else inputs))
    else:
        if not _LIGHTNING_SDK_AVAILABLE:
            raise ModuleNotFoundError("The `lightning_sdk` is required.")

        studio = Studio()
        job = studio._studio_api.create_data_prep_machine_job(
            f"cd {os.getcwd()} && python {sys.argv[0]}",
            name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            num_instances=num_nodes,
            studio_id=studio._studio.id,
            teamspace_id=studio._teamspace.id,
            cluster_id=studio._studio.cluster_id,
            cloud_compute=machine,
        )

        has_printed = False

        while True:
            curr_job = studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance(
                project_id=studio._teamspace.id, id=job.id
            )
            if curr_job.status.phase == "LIGHTNINGAPP_INSTANCE_STATE_FAILED":
                raise RuntimeError(f"job {curr_job.name} failed!")

            if curr_job.status.phase == "LIGHTNINGAPP_INSTANCE_STATE_STOPPED":
                break

            if not has_printed:
                cloud_url = get_lightning_cloud_url()
                job_url = f"{cloud_url}/{studio._owner.name}/{studio._teamspace.name}"
                job_url += f"/studios/{studio.name}/app?app_id=data-prep"
                print(f"Find your job at {job_url}")
                has_printed = True

            sleep(1)


def chunkify(
    fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    output_dir: str,
    chunk_size: Optional[int] = None,
    chunk_bytes: Optional[int] = None,
    compression: Optional[str] = None,
    name: Optional[str] = None,
    num_workers: Optional[int] = None,
    fast_dev_run: bool = False,
    num_nodes: Optional[int] = None,
    machine: Optional[str] = None,
) -> None:
    """This function converts a dataset into chunks possibly in a distributed way."""

    name = name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if chunk_size is None or chunk_bytes is None:
        raise ValueError("Either `chunk_size` or `chunk_bytes` needs to be defined.")

    if num_nodes is None or machine is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        data_processor = DataProcessor(
            name=name,
            num_workers=num_workers or os.cpu_count(),
            remote_output_dir=output_dir,
            fast_dev_run=fast_dev_run,
        )
        data_processor.run(
            LambdaDataChunkRecipe(
                fn,
                inputs() if callable(inputs) else inputs,
                chunk_size=chunk_size,
                chunk_bytes=chunk_bytes,
                compression=compression,
            )
        )
    else:
        if not _LIGHTNING_SDK_AVAILABLE:
            raise ModuleNotFoundError("The `lightning_sdk` is required.")

        studio = Studio()
        job = studio._studio_api.create_data_prep_machine_job(
            f"cd {os.getcwd()} && python {sys.argv[0]}",
            name=name,
            num_instances=num_nodes,
            studio_id=studio._studio.id,
            teamspace_id=studio._teamspace.id,
            cluster_id=studio._studio.cluster_id,
            cloud_compute=machine,
        )

        has_printed = False

        while True:
            curr_job = studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance(
                project_id=studio._teamspace.id, id=job.id
            )
            if curr_job.status.phase == "LIGHTNINGAPP_INSTANCE_STATE_FAILED":
                raise RuntimeError(f"job {curr_job.name} failed!")

            if curr_job.status.phase == "LIGHTNINGAPP_INSTANCE_STATE_STOPPED":
                break

            if not has_printed:
                cloud_url = get_lightning_cloud_url()
                job_url = f"{cloud_url}/{studio._owner.name}/{studio._teamspace.name}"
                job_url += f"/studios/{studio.name}/app?app_id=data-prep"
                print(f"Find your job at {job_url}")
                has_printed = True

            sleep(1)

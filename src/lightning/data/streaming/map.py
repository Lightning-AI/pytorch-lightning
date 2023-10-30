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

from time import sleep
import os
from typing import Any, Callable, List, Optional, Union
from datetime import datetime
from lightning.data.streaming.data_processor import DataProcessor, DataTransformRecipe
import sys


class LambdaDataTransformRecipe(DataTransformRecipe):
    def __init__(self, fn: Callable[[str, Any], None], inputs: List[Any]):
        super().__init__()
        self._fn = fn
        self._inputs = inputs

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, output_dir: str, item_metadata: Any) -> None:  # type: ignore
        self._fn(output_dir, item_metadata)


def map(
    fn: Callable[[str, Any], None],
    inputs: Union[Any, Callable],
    num_workers: Optional[int] = None,
    name: Optional[str] = None,
    remote_output_dir: Optional[str] = None,
    fast_dev_run: bool = False,
    version: int = 0,
    num_instances: Optional[int] = None,
    cloud_compute: Optional[str] = None,
) -> None:
    """This function executes a function over a collection of files possibly in a distributed way."""

    name = name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if num_instances is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        data_processor = DataProcessor(
            name=name,
            num_workers=num_workers or os.cpu_count(),
            remote_output_dir=remote_output_dir,
            fast_dev_run=fast_dev_run,
            version=version,
        )
        data_processor.run(LambdaDataTransformRecipe(fn, inputs() if callable(inputs) else inputs))
    else:
        from lightning_sdk import Machine, Studio

        studio = Studio()
        job = studio._studio_api.create_data_prep_machine_job(
            f"cd {os.getcwd()} && python {sys.argv[0]}",
            name=name,
            num_instances=num_instances,
            studio_id=studio._studio.id,
            teamspace_id=studio._teamspace.id,
            cluster_id=studio._studio.cluster_id,
            cloud_compute=cloud_compute,
        )

        while True:
            curr_job = studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance(project_id=studio._teamspace.id, id=job.id)
            if curr_job.status.phase == "LIGHTNINGAPP_INSTANCE_STATE_FAILED":
                raise RuntimeError(f"job {job_name} failed!")
            elif curr_job.status.phase == "LIGHTNINGAPP_INSTANCE_STATE_STOPPED":
                break

            print("Waiting")

            sleep(1)


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
import pickle
import shutil
import sys
from datetime import datetime
from typing import Optional

from lightning.app import _PROJECT_ROOT, LightningWork
from lightning.app.storage.path import _shared_local_mount_path
from lightning.app.utilities.imports import _is_docker_available, _is_jinja2_available, requires

if _is_docker_available():
    import docker
    from docker.models.containers import Container

if _is_jinja2_available():
    import jinja2


class DockerRunner:
    @requires("docker")
    def __init__(self, file: str, work: LightningWork, queue_id: str, create_base: bool = False):
        self.file = file
        self.work = work
        self.queue_id = queue_id
        self.image: Optional[str] = None
        if create_base:
            self._create_base_container()
        self._create_work_container()

    def _create_base_container(self) -> None:
        # 1. Get base container
        container_base = f"{_PROJECT_ROOT}/dockers/Dockerfile.base.cpu"
        destination_path = os.path.join(_PROJECT_ROOT, "Dockerfile")

        # 2. Copy the base Dockerfile within the Lightning project
        shutil.copy(container_base, destination_path)

        # 3. Build the docker image.
        os.system("docker build . --tag thomaschaton/base")

        # 4. Clean the copied base Dockerfile.
        os.remove(destination_path)

    def _create_work_container(self) -> None:
        # 1. Get work container.
        source_path = os.path.join(_PROJECT_ROOT, "dockers/Dockerfile.jinja")
        destination_path = os.path.join(_PROJECT_ROOT, "Dockerfile")
        work_destination_path = os.path.join(_PROJECT_ROOT, "work.p")

        # 2. Pickle the work.
        with open(work_destination_path, "wb") as f:
            pickle.dump(self.work, f)

        # 3. Load Lightning requirements.
        with open(source_path) as f:
            template = jinja2.Template(f.read())

        # Get the work local build spec.
        requirements = self.work.local_build_config.requirements

        # Render template with the requirements.
        dockerfile_str = template.render(
            requirements=" ".join(requirements),
            redis_host="host.docker.internal" if sys.platform == "darwin" else "127.0.0.1",
        )

        with open(destination_path, "w") as f:
            f.write(dockerfile_str)

        # 4. Build the container.
        self.image = f"work-{self.work.__class__.__qualname__.lower()}"
        os.system(f"docker build . --tag {self.image}")

        # 5. Clean the leftover files.
        os.remove(destination_path)
        os.remove(work_destination_path)

    def run(self) -> None:
        assert self.image

        # 1. Run the work container and launch the work.
        client = docker.DockerClient(base_url="unix://var/run/docker.sock")
        self.container: Container = client.containers.run(
            image=self.image,
            shm_size="10G",
            stderr=True,
            stdout=True,
            stdin_open=True,
            detach=True,
            ports=[url.split(":")[-1] for url in self.work._urls if url],
            volumes=[f"{str(_shared_local_mount_path())}:/home/.shared"],
            command=f"python -m lightning run work {self.file} --work_name={self.work.name} --queue_id {self.queue_id}",
            environment={"SHARED_MOUNT_DIRECTORY": "/home/.shared"},
            network_mode="host",
        )

        # 2. Check the log and exit when ``Starting WorkRunner`` is found.
        for line in self.container.logs(stream=True):
            line = str(line.strip())
            print(line)
            if "command not found" in line:
                raise RuntimeError("The container wasn't properly executed.")
            elif "Starting WorkRunner" in line:
                break

    def get_container_logs(self) -> str:
        """Returns the logs of the container produced until now."""
        return "".join([chr(c) for c in self.container.logs(until=datetime.now())])

    def kill(self) -> None:
        """Kill the container."""
        self.container.kill()

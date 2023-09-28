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

from lightning.data.cache.worker import get_worker_info


class _WorkerEnv:
    """Contains the environment for the current dataloader within the current training process.

    Args:
        world_size: The number of dataloader workers for the current training process
        rank: The rank of the current worker within the number of workers

    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    @classmethod
    def detect(cls) -> "_WorkerEnv":
        """Automatically detects the number of workers and the current rank.

        Note:
            This only works reliably within a dataloader worker as otherwise the necessary information won't be present.
            In such a case it will default to 1 worker

        """
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        current_worker_rank = worker_info.id if worker_info is not None else 0

        return cls(world_size=num_workers, rank=current_worker_rank)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(world_size: {self.world_size}, rank: {self.rank})"

    def __str__(self) -> str:
        return repr(self)

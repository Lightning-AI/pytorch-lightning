from typing import Optional

import torch
from torch.utils.data import get_worker_info


class _DistributedEnv:
    """The environment of the distributed training.

    Args:
        world_size: The number of total distributed training processes
        global_rank: The rank of the current process within this pool of training processes

    """

    def __init__(self, world_size: int, global_rank: int):
        self.world_size = world_size
        self.global_rank = global_rank

    @classmethod
    def detect(cls) -> "_DistributedEnv":
        """Tries to automatically detect the distributed environment paramters.

        Note:
            This detection may not work in processes spawned from the distributed processes (e.g. DataLoader workers)
            as the distributed framework won't be initialized there.
            It will default to 1 distributed process in this case.

        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        else:
            world_size = None
            global_rank = 0

        if world_size is None or world_size == -1:
            world_size = 1

        return cls(world_size=world_size, global_rank=global_rank)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(world_size: {self.world_size}, global_rank: {self.global_rank}\n)"

    def __str__(self) -> str:
        return repr(self)


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


class Environment:
    """Contains the compute environment. If not passed, will try to detect.

    Args:
        dist_env: The distributed environment (distributed worldsize and global rank)
        worker_env: The worker environment (number of workers, worker rank)

    """

    def __init__(self, dist_env: Optional[_DistributedEnv], worker_env: Optional[_WorkerEnv]):
        self.worker_env = worker_env
        self.dist_env = dist_env

    @classmethod
    def from_args(
        cls,
        dist_world_size: int,
        global_rank: int,
        num_workers: int,
        current_worker_rank: int,
    ) -> "Environment":
        """Generates the Environment class by already given arguments instead of detecting them.

        Args:
            dist_world_size: The worldsize used for distributed training (=total number of distributed processes)
            global_rank: The distributed global rank of the current process
            num_workers: The number of workers per distributed training process
            current_worker_rank: The rank of the current worker within the number of workers of
                the current training process

        """
        dist_env = _DistributedEnv(dist_world_size, global_rank)
        worker_env = _WorkerEnv(num_workers, current_worker_rank)
        return cls(dist_env=dist_env, worker_env=worker_env)

    @property
    def num_shards(self) -> int:
        """Returns the total number of shards.

        Note:
            This may not be accurate in a non-dataloader-worker process like the main training process
            as it doesn't necessarily know about the number of dataloader workers.

        """
        assert self.worker_env is not None
        assert self.dist_env is not None
        return self.worker_env.world_size * self.dist_env.world_size

    @property
    def shard_rank(self) -> int:
        """Returns the rank of the current process wrt. the total number of shards.

        Note:
            This may not be accurate in a non-dataloader-worker process like the main training process as it
            doesn't necessarily know about the number of dataloader workers.

        """
        assert self.worker_env is not None
        assert self.dist_env is not None
        return self.dist_env.global_rank * self.worker_env.world_size + self.worker_env.rank

    def __repr__(self) -> str:
        dist_env_repr = repr(self.dist_env)
        worker_env_repr = repr(self.worker_env)

        return (
            f"{self.__class__.__name__}(\n\tdist_env: {dist_env_repr},\n\tworker_env: "
            + f"{worker_env_repr}\n\tnum_shards: {self.num_shards},\n\tshard_rank: {self.shard_rank})"
        )

    def __str__(self) -> str:
        return repr(self)

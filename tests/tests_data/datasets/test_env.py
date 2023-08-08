from functools import partial

import pytest
import torch
from torch.utils.data import get_worker_info

from lightning.data.datasets.env import _DistributedEnv, _WorkerEnv, Environment
from lightning.fabric import Fabric
from tests_fabric.helpers.runif import RunIf


@pytest.mark.parametrize(
    (
        "num_workers",
        "current_worker_rank",
        "dist_world_size",
        "global_rank",
        "expected_num_shards",
        "expected_shard_rank",
    ),
    [
        pytest.param(1, 0, 1, 0, 1, 0),
        pytest.param(1, 0, 2, 0, 2, 0),
        pytest.param(1, 0, 2, 1, 2, 1),
        pytest.param(2, 0, 1, 0, 2, 0),
        pytest.param(2, 1, 1, 0, 2, 1),
        pytest.param(2, 0, 2, 0, 4, 0),
        pytest.param(2, 1, 2, 0, 4, 1),
        pytest.param(2, 0, 2, 1, 4, 2),
        pytest.param(2, 1, 2, 1, 4, 3),
    ],
)
def test_environment(
    num_workers,
    current_worker_rank,
    dist_world_size,
    global_rank,
    expected_num_shards,
    expected_shard_rank,
):
    env = Environment.from_args(dist_world_size, global_rank, num_workers, current_worker_rank)
    assert env.num_shards == expected_num_shards
    assert env.shard_rank == expected_shard_rank

    assert "Environment(" in repr(env)
    assert "Environment(" in str(env)

    assert "\n\tdist_env: _DistributedEnv(" in repr(env)
    assert "\n\tdist_env: _DistributedEnv(" in str(env)
    assert "_DistributedEnv(" in repr(env.dist_env)
    assert "_DistributedEnv(" in str(env.dist_env)

    assert "\n\tworker_env: _WorkerEnv(" in repr(env)
    assert "\n\tworker_env: _WorkerEnv(" in str(env)
    assert "_WorkerEnv(" in repr(env.worker_env)
    assert "_WorkerEnv(" in str(env.worker_env)

    assert f"world_size: {num_workers}" in repr(env)
    assert f"world_size: {num_workers}" in str(env)
    assert f"world_size: {num_workers}" in repr(env.worker_env)
    assert f"world_size: {num_workers}" in str(env.worker_env)

    assert f"rank: {current_worker_rank}" in repr(env)
    assert f"rank: {current_worker_rank}" in str(env)
    assert f"rank: {current_worker_rank}" in repr(env.worker_env)
    assert f"rank: {current_worker_rank}" in str(env.worker_env)

    assert f"world_size: {dist_world_size}" in repr(env)
    assert f"world_size: {dist_world_size}" in str(env)
    assert f"world_size: {dist_world_size}" in repr(env.dist_env)
    assert f"world_size: {dist_world_size}" in str(env.dist_env)

    assert f"global_rank: {global_rank}" in repr(env)
    assert f"global_rank: {global_rank}" in str(env)
    assert f"global_rank: {global_rank}" in repr(env.dist_env)
    assert f"global_rank: {global_rank}" in str(env.dist_env)

    assert f"shard_rank: {expected_shard_rank}" in repr(env)
    assert f"shard_rank: {expected_shard_rank}" in str(env)

    assert f"num_shards: {expected_num_shards}" in repr(env)
    assert f"num_shards: {expected_num_shards}" in str(env)


class EnvTestDataset(torch.utils.data.IterableDataset):
    def __init__(self, num_workers, dist_size, global_rank):
        self.num_workers = num_workers
        self.dist_size = dist_size
        self.global_rank = global_rank
        self.env = Environment(_DistributedEnv.detect(), None)

    def __iter__(self):
        worker_info = get_worker_info()
        env = self.env
        env.worker_env = _WorkerEnv.detect()
        assert env.worker_env.world_size == self.num_workers
        assert env.dist_env.world_size == self.dist_size
        assert env.dist_env.global_rank == self.global_rank
        assert env.worker_env.rank == (worker_info.id if worker_info is not None else 0)

        yield 0


def env_auto_test(fabric: Fabric, num_workers):
    dset = EnvTestDataset(max(1, num_workers), fabric.world_size, fabric.global_rank)
    loader = torch.utils.data.DataLoader(dset, num_workers=num_workers)

    # this triggers the `__iter__` of the dataset containing the actual test
    for _ in loader:
        pass


@RunIf(skip_windows=True)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("dist_world_size", [1, 2])
def test_env_auto(num_workers, dist_world_size):
    fabric = Fabric(accelerator="cpu", devices=dist_world_size, strategy="ddp_spawn")
    fabric.launch(partial(env_auto_test, num_workers=num_workers))

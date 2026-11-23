"""Local experiment for changing just the rendezvous used by this test."""

import time
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch.distributed as dist

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.strategies import DDPStrategy


class FileRendezvousStrategy(DDPStrategy):
    rendezvous_path: str

    def setup_distributed(self):
        def init(cluster_environment, backend, *, timeout, device_id):
            store = dist.FileStore(self.rendezvous_path, cluster_environment.world_size())
            dist.init_process_group(
                backend,
                store=store,
                rank=cluster_environment.global_rank(),
                world_size=cluster_environment.world_size(),
                timeout=timeout,
                device_id=device_id,
            )

        with patch("lightning.pytorch.strategies.ddp._init_dist_connection", init):
            super().setup_distributed()


@pytest.mark.parametrize("store", ["tcp", "file"])
def test_distributed_profiler_all_stages(tmp_path, store):
    strategy_cls = DDPStrategy if store == "tcp" else FileRendezvousStrategy
    strategy = strategy_cls(start_method="spawn", timeout=timedelta(seconds=90))
    reports = tmp_path / "reports"
    profiler = SimpleProfiler(dirpath=reports, filename="profiler")
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=2,
        strategy=strategy,
        accelerator="cpu",
        devices=2,
        profiler=profiler,
        logger=False,
    )
    model = BoringModel()

    for name, run in (("fit", trainer.fit), ("validate", trainer.validate), ("test", trainer.test)):
        if store == "file":
            strategy.rendezvous_path = str(tmp_path / name)
        start = time.perf_counter()
        run(model)
        print(f"{store} {name}: {time.perf_counter() - start:.3f}s")

    expected = {f"{stage}-profiler-{rank}.txt" for stage in ("fit", "validate", "test") for rank in (0, 1)}
    assert {path.name for path in reports.iterdir()} == expected
    assert all(path.read_text("utf-8") for path in reports.iterdir())

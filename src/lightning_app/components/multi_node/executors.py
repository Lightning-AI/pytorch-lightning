import os

from lightning_app.utilities.proxies import WorkRunExecutor


class LiteRunExecutor(WorkRunExecutor):
    def __call__(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ):
        os.environ["MASTER_ADDR"] = main_address
        os.environ["MASTER_PORT"] = str(main_port)
        os.environ["NODE_RANK"] = str(node_rank)

        lite = self.configure_lite(num_nodes)
        lite.launch(function=self.work_run)

    def configure_lite(self, num_nodes: int):
        from lightning.lite import LightningLite

        return LightningLite(accelerator="auto", devices="auto", strategy="ddp_spawn", num_nodes=num_nodes)


class PyTorchSpawnRunExecutor(WorkRunExecutor):
    def __call__(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ):
        import torch

        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        torch.multiprocessing.spawn(
            self.run, args=(main_address, main_port, num_nodes, node_rank, nprocs), nprocs=nprocs
        )

    def run(self, local_rank: int, main_address: str, main_port: int, num_nodes: int, node_rank: int, nprocs: int):
        import torch

        # 1. Setting distributed environment
        global_rank = local_rank + node_rank * nprocs
        world_size = num_nodes * nprocs

        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl" if torch.cuda.is_available() else "gloo",
                rank=global_rank,
                world_size=world_size,
                init_method=f"tcp://{main_address}:{main_port}",
            )

        self.work_run(world_size, node_rank, global_rank, local_rank)

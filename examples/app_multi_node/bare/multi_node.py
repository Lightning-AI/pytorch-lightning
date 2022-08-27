import lightning as L


class Work(L.LightningWork):
    def __init__(self, cloud_compute: L.CloudCompute = L.CloudCompute(), **kwargs):
        super().__init__(parallel=True, **kwargs, cloud_compute=cloud_compute)

    def run(self, main_address="localhost", main_port=1111, world_size=1, rank=0, init=False):
        if init:
            return

        import torch.distributed

        print(f"Initializing process group: {main_address=}, {main_port=}, {world_size=}, {rank=}")
        torch.distributed.init_process_group(
            backend="gloo", init_method=f"tcp://{main_address}:{main_port}", world_size=world_size, rank=rank
        )
        gathered = [torch.zeros(1) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, torch.tensor([rank]).float())
        print(gathered)


class MultiNodeDemo(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.work0 = Work()
        self.work1 = Work()

    def run(self):
        self.work0.run(init=True)
        if self.work0.internal_ip:
            self.work0.run(main_address=self.work0.internal_ip, main_port=self.work0.port, world_size=2, rank=0)
            self.work1.run(main_address=self.work0.internal_ip, main_port=self.work0.port, world_size=2, rank=1)


app = L.LightningApp(MultiNodeDemo())

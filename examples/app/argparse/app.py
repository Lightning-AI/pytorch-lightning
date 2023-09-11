import argparse

from lightning.app import CloudCompute, LightningApp, LightningFlow, LightningWork


class Work(LightningWork):
    def __init__(self, cloud_compute):
        super().__init__(cloud_compute=cloud_compute)

    def run(self):
        pass


class Flow(LightningFlow):
    def __init__(self, cloud_compute):
        super().__init__()
        self.work = Work(cloud_compute)

    def run(self):
        assert self.work.cloud_compute.name == "gpu", self.work.cloud_compute.name
        self.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Whether to use GPU in the cloud")
    hparams = parser.parse_args()
    app = LightningApp(Flow(CloudCompute("gpu" if hparams.use_gpu else "cpu")))

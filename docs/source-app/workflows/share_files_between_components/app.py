import os

import torch

from lightning.app import LightningWork, LightningFlow, LightningApp
from lightning.app.storage.path import Path


class ModelTraining(LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoints_path = Path("./checkpoints")

    def run(self):
        # make fake checkpoints
        checkpoint_1 = torch.tensor([0, 1, 2, 3, 4])
        checkpoint_2 = torch.tensor([0, 1, 2, 3, 4])
        os.makedirs(self.checkpoints_path, exist_ok=True)
        checkpoint_path = str(self.checkpoints_path / "checkpoint_{}.ckpt")
        torch.save(checkpoint_1, str(checkpoint_path).format("1"))
        torch.save(checkpoint_2, str(checkpoint_path).format("2"))


class ModelDeploy(LightningWork):
    def __init__(self, ckpt_path, *args, **kwargs):
        super().__init__()
        self.ckpt_path = ckpt_path

    def run(self):
        ckpts = os.listdir(self.ckpt_path)
        checkpoint_1 = torch.load(os.path.join(self.ckpt_path, ckpts[0]))
        checkpoint_2 = torch.load(os.path.join(self.ckpt_path, ckpts[1]))
        print(f"Loaded checkpoint_1: {checkpoint_1}")
        print(f"Loaded checkpoint_2: {checkpoint_2}")


class LitApp(LightningFlow):
    def __init__(self):
        super().__init__()
        self.train = ModelTraining()
        self.deploy = ModelDeploy(ckpt_path=self.train.checkpoints_path)

    def run(self):
        self.train.run()
        self.deploy.run()


app = LightningApp(LitApp())

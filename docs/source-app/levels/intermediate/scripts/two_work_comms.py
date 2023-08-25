# app.py
from lightning.app import LightningWork, LightningFlow, LightningApp
import time

class TrainComponent(LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_checkpoint_path = None

    def run(self):
        # pretend to train and save a checkpoint every 10 steps
        for step in (range(1000)):
            time.sleep(1.0)
            fake_loss = round(1/(step + 0.00001), 4)
            print(f'{step=}: {fake_loss=} ')
            if step % 10 == 0:
                self.last_checkpoint_path = f'/some/path/{step=}_{fake_loss=}'
                print(f'TRAIN COMPONENT: saved new checkpoint: {self.last_checkpoint_path}')

class ModelDeploymentComponent(LightningWork):
    def run(self, new_checkpoint):
        print(f'DEPLOY COMPONENT: load new model from checkpoint: {new_checkpoint}')

class ContinuousDeployment(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train = TrainComponent(parallel=True)
        self.model_deployment = ModelDeploymentComponent(parallel=True)

    def run(self):
        self.train.run()
        if self.train.last_checkpoint_path:
            self.model_deployment.run(self.train.last_checkpoint_path)

app = LightningApp(ContinuousDeployment())

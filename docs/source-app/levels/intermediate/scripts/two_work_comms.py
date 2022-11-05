# app.py
import lightning as L
from lightning.app.perf import debug
import time

class TrainComponent(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_checkpoint_path = ''
        self.has_new_checkpoint = False

    def run(self):
        # pretend to train and save a checkpoint every 10 steps
        for step in (range(1000)):
            time.sleep(1.0)
            fake_loss = 1/(step + 0.00001)
            print(f'{step=}: {fake_loss=} ')
            if step % 10 == 0:
                self.last_checkpoint_path = f'/some/path/{step=}_{fake_loss=}'
                print(f'saved new checkpoint: {self.last_checkpoint_path}')
                debug.set_trace()
                self.has_new_checkpoint = True

class ModelDeploymentComponent(L.LightningWork):
    def run(self, new_checkpoint):
        print(f'load new model from checkpoint: {new_checkpoint}')

class ContinuousDeployment(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train = TrainComponent(parallel=True)
        self.model_deployment = ModelDeploymentComponent(parallel=True)

    def run(self):
        self.train.run()
        if self.train.has_new_checkpoint:
            self.train.has_new_checkpoint = False
            print(f'new ckpt: {self.train.last_checkpoint_path}')
            self.model_deployment.run(self.train.last_checkpoint_path)

app = L.LightningApp(ContinuousDeployment())
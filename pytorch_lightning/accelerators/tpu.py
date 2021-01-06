# TODO: Complete the TPUAccelerator
from pytorch_lightning.accelerators.accelerator import Accelerator


class TPUAccelerator(Accelerator):
    def setup(self, trainer, model):
        raise NotImplementedError

    def on_train_start(self):
        raise NotImplementedError

    def on_train_end(self):
        raise NotImplementedError

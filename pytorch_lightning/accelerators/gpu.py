import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class GPUAccelerator(Accelerator):

    def setup(self, trainer, model):
        if "cuda" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be GPU, got {self.root_device} instead")
        torch.cuda.set_device(self.root_device)
        model.to(self.root_device)

        return super().setup(trainer, model)

    def on_train_start(self):
        # clear cache before training
        # use context because of:
        # https://discuss.pytorch.org/t/out-of-memory-when-i-use-torch-cuda-empty-cache/57898
        with torch.cuda.device(self.root_device):
            torch.cuda.empty_cache()

    def on_train_end(self):
        # clean up memory
        with torch.cuda.device(self.root_device):
            torch.cuda.empty_cache()

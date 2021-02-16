import logging
import os

import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)


class GPUAccelerator(Accelerator):

    def setup(self, trainer, model):
        if "cuda" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be GPU, got {self.root_device} instead")
        self.set_nvidia_flags()
        torch.cuda.set_device(self.root_device)
        return super().setup(trainer, model)

    def on_train_start(self):
        # clear cache before training
        # use context because of:
        # https://discuss.pytorch.org/t/out-of-memory-when-i-use-torch-cuda-empty-cache/57898
        with torch.cuda.device(self.root_device):
            torch.cuda.empty_cache()

    def on_train_end(self):
        # clean up memory
        self.model.cpu()
        with torch.cuda.device(self.root_device):
            torch.cuda.empty_cache()

    @staticmethod
    def set_nvidia_flags():
        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join([str(x) for x in range(torch.cuda.device_count())])
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        _log.info(f"LOCAL_RANK: {os.getenv('LOCAL_RANK', 0)} - CUDA_VISIBLE_DEVICES: [{devices}]")

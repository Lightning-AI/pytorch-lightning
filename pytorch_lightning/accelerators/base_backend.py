import torch
from typing import Any
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.trainer import Trainer


class Accelerator(object):

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def batch_to_device(self, batch: Any, device: torch.device):
        model = self.trainer.get_model()
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)

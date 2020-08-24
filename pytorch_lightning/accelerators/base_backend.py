import torch
from typing import Any
from pytorch_lightning.utilities.apply_func import move_data_to_device


class Accelerator(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def batch_to_device(self, batch: Any, device: torch.device):
        model = self.trainer.get_model()
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)

    def on_test_batch_end(self, output):
        return output

    def on_validation_batch_end(self, output):
        return output

    def training_step_end(self, output):
        return output

    def test_step_end(self, output):
        return output

    def validation_step_end(self, output):
        return output

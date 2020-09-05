import torch
from typing import Any
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities import AMPType, rank_zero_warn


class Accelerator(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def setup(self, model):
        pass

    def teardown(self):
        pass

    def batch_to_device(self, batch: Any, device: torch.device):
        model = self.trainer.get_model()
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)

    def training_step_end(self, output):
        return output

    def test_step_end(self, output):
        return output

    def validation_step_end(self, output):
        return output

    def process_dataloader(self, dataloader):
        return dataloader

    def backward(self, closure_loss, optimizer, opt_idx):
        model_ref = self.trainer.get_model()

        # scale loss for 16 bit
        if self.trainer.precision == 16:
            closure_loss = model_ref.amp_scale_loss(
                closure_loss,
                optimizer,
                opt_idx,
                amp_backend=self.trainer.amp_backend
            )

            # enter amp context
            if self.trainer.amp_backend == AMPType.APEX:
                self.trainer.dev_debugger.track_event('AMP', str(AMPType.APEX))
                context = closure_loss
                closure_loss = closure_loss.__enter__()

        # do backward pass
        model_ref.backward(self, closure_loss, optimizer, opt_idx)

        # exit amp context
        if self.trainer.precision == 16 and self.trainer.amp_backend == AMPType.APEX:
            a, b, c = None, None, None
            error = context.__exit__(a, b, c)
            if error:
                rank_zero_warn(a, b, c)
                raise Exception('apex unscale error')

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()
        return closure_loss

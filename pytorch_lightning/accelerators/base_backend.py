import torch
from typing import Any
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


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

    def optimizer_step(self, optimizer, batch_idx, opt_idx, lambda_closure):
        model_ref = self.trainer.get_model()
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        native_amp = self.trainer.amp_backend == AMPType.NATIVE

        # native amp + lbfgs is a no go right now
        if native_amp and is_lbfgs:
            raise MisconfigurationException(
                'native PyTorch amp and lbfgs are not compatible.'
                ' To request, please file a Github issue in PyTorch and tag @mcarilli')

        # model hook
        model_ref.optimizer_step(
            self.trainer.current_epoch,
            batch_idx,
            optimizer,
            opt_idx,
            lambda_closure,
            using_native_amp=native_amp,
            using_lbfgs=is_lbfgs
        )

        # scale when native amp
        if native_amp:
            self.trainer.scaler.update()

    def optimizer_zero_grad(self, batch_idx, optimizer, opt_idx):
        model_ref = self.trainer.get_model()
        model_ref.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)

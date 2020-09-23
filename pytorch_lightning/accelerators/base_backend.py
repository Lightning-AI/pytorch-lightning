import torch
from typing import Any
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import math


try:
    from apex import amp
except ImportError:
    amp = None

EPSILON = 1e-6
EPSILON_FP16 = 1e-5


class Accelerator(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def setup(self, model):
        pass

    def teardown(self):
        pass

    def barrier(self, name: str = None):
        pass

    def train_or_test(self):
        if self.trainer.testing:
            results = self.trainer.run_test()
        else:
            results = self.trainer.train()
        return results

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

    def clip_gradients(self, optimizer):

        if self.trainer.amp_backend == AMPType.NATIVE:
            self.trainer.scaler.unscale_(optimizer)

        # apply clip gradients
        # TODO: separate TPU case from here
        self._clip_gradients(optimizer)

    def _clip_gradients(self, optimizer):
        # this code is a modification of torch.nn.utils.clip_grad_norm_
        # with TPU support based on https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md
        if self.trainer.gradient_clip_val <= 0:
            return

        model = self.trainer.get_model()
        if self.trainer.amp_backend == AMPType.APEX:
            parameters = amp.master_params(optimizer)
        else:
            parameters = model.parameters()

        max_norm = float(self.trainer.gradient_clip_val)
        norm_type = float(2.0)

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        if norm_type == math.inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            device = parameters[0].device
            out = torch.empty(len(parameters), device=device)
            for i, p in enumerate(parameters):
                torch.norm(p.grad.data.to(device), norm_type, out=out[i])
            total_norm = torch.norm(out, norm_type)

        eps = EPSILON_FP16 if self.trainer.precision == 16 else EPSILON
        clip_coef = torch.tensor(max_norm, device=device) / (total_norm + eps)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        for p in parameters:
            p.grad.data.mul_(clip_coef.to(p.grad.data.device))

    def on_train_epoch_end(self):
        pass

    def early_stopping_should_stop(self, pl_module):
        return self.trainer.should_stop

from pytorch_lightning.accelerators.data_parallel import ParallelPlugin, TrainingTypePlugin
from pytorch_lightning.accelerators.base_plugin import Plugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import NATIVE_AMP_AVAILABLE, AMPType
from typing import Any, Union
import math

import torch
from torch.optim import Optimizer

from pytorch_lightning.core import LightningModule
from pytorch_lightning.accelerators.precision import (
    ApexMixedPrecisionPlugin,
    MixedPrecisionPlugin,
    NativeMixedPrecisionPlugin,
    PrecisionPlugin,
)

from pytorch_lightning.utilities.apply_func import move_data_to_device


class NewAccelerator(object):
    def __init__(
        self,
        precision_plugin: PrecisionPlugin,
        training_type_plugin: TrainingTypePlugin,
    ):
        self.precision_plugin = precision_plugin
        self.training_type_plugin = training_type_plugin

        self.optimizers = None
        self.lr_schedulers = None
        self.optimizer_frequencies = None

    def setup(self, trainer, model):
        self.connect_training_type_plugin(self.training_type_plugin, model)
        self.setup_optimizers(trainer, model)
        self.connect_precision_plugin(self.precision_plugin)

    @property
    def model(self):
        return self.training_type_plugin.model

    @model.setter
    def model(self, new_model):
        self.training_type_plugin.model = new_model

    @property
    def lightning_module(self):
        return self.training_type_plugin.lightning_module

    @property
    def root_device(self):
        return self.training_type_plugin.root_device

    def teardown(self):
        pass

    def batch_to_device(self, batch: Any, device: torch.device):
        model = self.lightning_module
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)

    def on_train_start(self):
        pass

    def training_step(self, args):
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.train_step_context():
            with self.training_type_plugin.train_step_context():
                return self.lightning_module.training_step(*args)

    def validation_step(self, args):
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.val_step_context():
            with self.training_type_plugin.val_step_context():
                return self.lightning_module.validation_step(*args)

    def test_step(self, args):
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.test_step_context():
            with self.training_type_plugin.test_step_context():
                return self.lightning_module.test_step(*args)

    def training_step_end(self, output):
        return output

    def test_step_end(self, output):
        return output

    def validation_step_end(self, output):
        return output

    def process_dataloader(self, dataloader):
        return dataloader

    def backward(self, closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs):
        return self.precision_plugin.backward(
            self.lightning_module, closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs
        )

    def optimizer_step(self, optimizer, current_epoch, batch_idx, opt_idx, lambda_closure):

        model_ref = self.lightning_module
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        native_amp = (
            isinstance(self.precision_plugin, MixedPrecisionPlugin) and self.precision_plugin.backend == AMPType.NATIVE
        )

        self.precision_plugin.pre_optimizer_step(optimizer, opt_idx)

        # model hook
        res = model_ref.optimizer_step(
            epoch=current_epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=opt_idx,
            optimizer_closure=lambda_closure,
            on_tpu=False,  # TPUAccelerator class sets this as True
            using_native_amp=native_amp,
            using_lbfgs=is_lbfgs,
        )

        self.precision_plugin.post_optimizer_step(optimizer, opt_idx)
        return res

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
        model_ref = self.lightning_module
        model_ref.optimizer_zero_grad(current_epoch, batch_idx, optimizer, opt_idx)

    def clip_gradients(self, optimizer, clip_val):
        # TODO: separate TPU case from here
        self._clip_gradients(optimizer, clip_val)

    def _clip_gradients(self, optimizer, grad_clip_val):
        if grad_clip_val is None:
            return

        grad_clip_val = float(grad_clip_val)

        if grad_clip_val <= 0:
            return
        self._clip_gradients(optimizer, grad_clip_val)

        model = self.lightning_module

        # TODO: Change this. Probably to isinstance(self.precision_plugin, MixedPrecisionPlugin) and self.precision_plugin.backend == AMPType.APEX
        #  ... or we call master_params() and in the default plugin we return the model.parameters()
        if self.trainer.amp_backend == AMPType.APEX:
            parameters = self.precision_plugin.master_params(optimizer)
        else:
            parameters = model.parameters()

        max_norm = grad_clip_val
        norm_type = float(2.0)

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        device = parameters[0].device

        if norm_type == math.inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            out = torch.empty(len(parameters), device=device)
            for i, p in enumerate(parameters):
                torch.norm(p.grad.data.to(device), norm_type, out=out[i])
            total_norm = torch.norm(out, norm_type)

        eps = self.precision_plugin.EPSILON

        clip_coef = torch.tensor(max_norm, device=device) / (total_norm + eps)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        for p in parameters:
            p.grad.data.mul_(clip_coef.to(p.grad.data.device))

    def on_train_epoch_end(self, outputs):
        pass

    def on_train_end(self):
        pass

    # TODO: Check if we can change logic for early stopping to accelerator/trainer completely or have a separate connector (should be self contained)
    def early_stopping_should_stop(self, pl_module):
        return self.trainer.should_stop

    def setup_optimizers(self, trainer, model):
        if trainer.testing is True:
            return
        optimizers, lr_schedulers, optimizer_frequencies = trainer.init_optimizers(model)
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.optimizer_frequencies = optimizer_frequencies

    def connect_training_type_plugin(self, plugin: TrainingTypePlugin, model: LightningModule):
        plugin.connect(model)

    def connect_precision_plugin(self, plugin: PrecisionPlugin):
        model, optimizers, schedulers = plugin.connect(self.model, self.optimizers, self.lr_schedulers)

        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers

    def to_device(self, batch):
        return self.batch_to_device(batch, self.root_device)

    @property
    def amp_backend(self):
        if isinstance(self.precision_plugin, ApexMixedPrecisionPlugin):
            return AMPType.APEX
        elif isinstance(self.precision_plugin, NativeMixedPrecisionPlugin):
            return AMPType.NATIVE
        else:
            return None

    @property
    def precision(self):
        return self.precision_plugin.precision

    @property
    def scaler(self):
        if hasattr(self.precision_plugin, "scaler"):
            return self.precision_plugin.scaler

        return None

    @property
    def rpc_enabled(self):
        return self.training_type_plugin.rpc_enabled

    # TODO: Check where this comes from and why it is needed
    def optimizer_state(self, optimizer: Optimizer) -> dict:
        """
        Returns state of an optimizer. Allows for syncing/collating optimizer state from processes in custom
        plugins.
        Return:
            Optimizer state dict
        """
        if self.training_type_plugin and hasattr(self.training_type_plugin, "optimizer_state"):
            return self.training_type_plugin.optimizer_state(optimizer)
        return optimizer.state_dict()

    def on_save(self, checkpoint):
        return checkpoint


class NewCPUAccelerator(NewAccelerator):
    def setup(self, trainer, model):
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            MisconfigurationException("amp + cpu is not supported.  Please use a GPU option")

        if "cpu" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be CPU, got {self.root_device} instead")

        return super().setup(trainer, model)


class NewGPUAccelerator(NewAccelerator):
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


# TODO: Complete the TPUAccelerator
class NewTPUAccelerator(NewAccelerator):
    def setup(self, trainer, model):
        raise NotImplementedError

    def on_train_start(self):
        raise NotImplementedError

# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, cast

import torch

from pytorch_lightning.accelerators import DDPAccelerator
from pytorch_lightning.overrides.fairscale import LightningOSS, LightningShardedDataParallel
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class DDPShardedAccelerator(DDPAccelerator):

    def __init__(self, trainer, cluster_environment=None, ddp_plugin=None):
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.nickname = 'ddp_sharded'

    def setup_optimizers(self, model):
        if self.trainer.testing is True:
            return

        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = self.re_init_with_fairscale_zero(optimizers, lr_schedulers)
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

    def re_init_with_fairscale_zero(self, optimizers, lr_schedulers):
        """
        Re-initialise optimizers to use OSS wrapper. We need to re-initialise due to
        the parameters being sharded across distributed processes, each optimizing a partition.
        Args:
            optimizers: Input optimizers for trainer.
        Returns: Optimizers re-initialised using FairScale OSS (ZERO optimizer).

        """
        fairscale_zero_optimizers = []
        for optimizer in optimizers:
            if not isinstance(optimizer, LightningOSS):
                optim_class = type(optimizer)
                zero_optimizer = LightningOSS(
                    params=optimizer.param_groups,
                    optim=optim_class,
                    **optimizer.defaults
                )
                fairscale_zero_optimizers.append(zero_optimizer)
                for scheduler in lr_schedulers:
                    scheduler = scheduler['scheduler']
                    if scheduler.optimizer == optimizer:
                        scheduler.optimizer = zero_optimizer
                del optimizer
        return fairscale_zero_optimizers

    def sync_optim_state(self):
        for optimizer in self.trainer.optimizers:
            optimizer.consolidate_state_dict()

    def configure_ddp(
            self, model: "LightningModule", device_ids: List[int]
    ):
        if self.trainer.model.testing:  # Stick to standard DDP if using testing
            model = self.ddp_plugin.configure_ddp(model, device_ids)
        else:
            model = LightningShardedDataParallel(model, sharded_optimizer=self.trainer.optimizers)
        return model

    def training_step(self, args):
        return self._trainer_step(args)

    def validation_step(self, args):
        return self._trainer_step(args)

    def test_step(self, args):
        return self._trainer_step(args)

    def _trainer_step(self, args):
        if self.trainer.on_gpu:
            batch = args[0]
            batch = self.batch_to_device(batch, self.trainer.root_gpu)
            args[0] = batch

        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.trainer.model(*args)
        else:
            output = self.trainer.model(*args)
        return output

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        closure_loss = super().backward(
            closure_loss=closure_loss,
            optimizer=optimizer,
            opt_idx=opt_idx,
            *args,
            **kwargs
        )
        # Ensure all backward handles have been called before calling optimizer step
        self.trainer.model = cast(LightningShardedDataParallel, self.trainer.model)
        self.trainer.model.clear_backward_handles()
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
            epoch=self.trainer.current_epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=opt_idx,
            optimizer_closure=lambda_closure,
            on_tpu=False,  # TPUAccelerator class sets this as True
            using_native_amp=native_amp,
            using_lbfgs=is_lbfgs
        )

        if self.trainer.amp_backend == AMPType.NATIVE:
            self._broadcast_weights_if_optimizer_step_skipped(optimizer)
            # scale when native amp
            self.trainer.scaler.update()

    def _broadcast_weights_if_optimizer_step_skipped(self, optimizer):
        scaler = self.trainer.scaler

        optimizer_state = scaler._per_optimizer_states[id(optimizer)]
        infs_detected = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())
        if infs_detected:
            # Sync all the updated shards in between the ranks
            for (device,
                 device_params,) in optimizer.per_device_params.items():  # all the params on this device (inc all ranks)
                optimizer._broadcast_params(optimizer._broadcast_buffers[device], device_params)

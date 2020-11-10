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
import io
import os
import re
from typing import Optional

import torch
import torch.multiprocessing as mp

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils

TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()

if TPU_AVAILABLE:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as xla_pl
    import torch_xla.distributed.xla_multiprocessing as xmp


class TPUAccelerator(Accelerator):

    def __init__(self, trainer, cluster_environment=None):
        """
        Runs training using TPUs (colab, single machine or pod)

        Example::

            # default
            trainer = Trainer(accelerator=TPUAccelerator())

        """
        super().__init__(trainer, cluster_environment)
        self.start_method = None
        self.mp_queue = None
        self.nickname = None

    def setup(self, model):
        rank_zero_info(f'training on {self.trainer.tpu_cores} TPU cores')

        # TODO: Move this check to Trainer __init__ or device parser
        if not TPU_AVAILABLE:
            raise MisconfigurationException('PyTorch XLA not installed.')

        # see: https://discuss.pytorch.org/t/segfault-with-multiprocessing-queue/81292/2
        self.start_method = 'fork'

        # pass in a state q
        smp = mp.get_context(self.start_method)
        self.mp_queue = smp.SimpleQueue()

        self.trainer.model = model

    def teardown(self):
        model = self.trainer.model

        # restore main state with best weights
        best_path = self.mp_queue.get()
        results = self.mp_queue.get()
        last_path = self.mp_queue.get()

        # transfer back the best path to the trainer
        if self.trainer.checkpoint_callback is not None:
            self.trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also bets score

        # load last weights
        if last_path and not self.trainer.testing:
            ckpt = torch.load(last_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt)

        self.trainer.model = model

        # when training completes, load the weights back in main process
        self.__load_weights_on_main_process()
        return results

    def train(self):
        model = self.trainer.model

        # train
        if self.trainer.tpu_id is not None:
            self.tpu_train_in_process(self.trainer.tpu_id, model, self.trainer, self.mp_queue)
        else:
            xmp.spawn(
                self.tpu_train_in_process,
                args=(model, self.trainer, self.mp_queue),
                nprocs=self.trainer.tpu_cores,
                start_method=self.start_method
            )

    def __load_weights_on_main_process(self):
        model = self.trainer.model

        # load weights if not interrupted
        if self.trainer.on_colab_kaggle and not self.trainer.testing:
            self.load_spawn_weights(model)

        self.trainer.model = model

    def tpu_train_in_process(self, tpu_core_idx: int, model: LightningModule, trainer=None, mp_queue=None):
        """
        Here we are inside each individual process
        """
        if not trainer:
            trainer = self.trainer

        trainer.call_setup_hook(model)

        # setup TPU training
        self.__setup_tpu_training(model, trainer)

        # set up training routine
        self.trainer.train_loop.setup_training(model)

        # train or test
        results = self.train_or_test()

        # save weights at the end of training
        self.__save_end_of_training_weights(model, trainer)

        # persist info in spawn
        self.transfer_distrib_spawn_state_on_fit_end(model, mp_queue, results)

    def training_step(self, args):
        batch = args[0]
        batch = self.to_device(batch)
        args[0] = batch
        output = self.trainer.model.training_step(*args)
        return output

    def validation_step(self, args):
        batch = args[0]
        batch = self.to_device(batch)
        args[0] = batch
        output = self.trainer.model.validation_step(*args)
        return output

    def test_step(self, args):
        batch = args[0]
        batch = self.to_device(batch)
        args[0] = batch
        output = self.trainer.model.test_step(*args)
        return output

    def process_dataloader(self, dataloader):
        device = xm.xla_device(self.trainer.tpu_id)
        dataloader = xla_pl.ParallelLoader(dataloader, [device])
        dataloader = dataloader.per_device_loader(device)
        return dataloader

    def to_device(self, batch):
        """
        Transfers the data to the TPU.

        Args:
            batch: A tensor or collection of tensors.
            tpu_id: The id of the TPU core. If omitted, the first available core is chosen.

        Return:
            the tensor on the TPU device.

        See Also:
            - :func:`~pytorch_lightning.utilities.apply_func.move_data_to_device`
        """
        if not TPU_AVAILABLE:
            raise MisconfigurationException(
                'Requested to transfer batch to TPU but XLA is not available.'
                ' Are you sure this machine has TPUs?'
            )
        device = xm.xla_device(self.trainer.tpu_id)

        return self.batch_to_device(batch, device)

    def __save_end_of_training_weights(self, model: LightningModule, trainer):
        # when training ends on these platforms dump weights to get out of the main process
        if trainer.on_colab_kaggle:
            rank_zero_warn('cleaning up... please do not interrupt')
            self.save_spawn_weights(model)

    def __setup_tpu_training(self, model: LightningModule, trainer):
        # use the default device from the process
        # tpu_device = xm.xla_device()

        # if given an ordinal device, use this as the device
        if trainer.tpu_id is not None:
            tpu_device = xm.xla_device(trainer.tpu_id)
        else:
            tpu_device = xm.xla_device()
        # track the device and move model to it
        trainer._device = tpu_device
        model.to(trainer._device)

        # get the appropriate tpu ranks
        trainer.tpu_local_core_rank = xm.get_local_ordinal()
        trainer.tpu_global_core_rank = xm.get_ordinal()

        # avoid duplicating progress bar
        if trainer.tpu_global_core_rank != 0 and trainer.progress_bar_callback is not None:
            trainer.progress_bar_callback.disable()

        trainer.global_rank = trainer.tpu_local_core_rank
        rank_zero_only.rank = trainer.global_rank

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        # init 16 bit for TPU
        if trainer.precision == 16:
            os.environ['XLA_USE_BF16'] = str(1)

        log.info(f'INIT TPU local core: {trainer.tpu_local_core_rank},'
                 f' global rank: {trainer.tpu_global_core_rank}'
                 f' with XLA_USE_BF16={os.environ.get("XLA_USE_BF16")}')

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        # do backward pass
        if self.trainer.train_loop.automatic_optimization:
            model = self.trainer.get_model()
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # detach after backward
        closure_loss = closure_loss.detach()

        return closure_loss

    def optimizer_step(self, optimizer, batch_idx, opt_idx, lambda_closure):
        model_ref = self.trainer.get_model()
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

        # model hook
        model_ref.optimizer_step(
            epoch=self.trainer.current_epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=opt_idx,
            optimizer_closure=lambda_closure,
            on_tpu=True,
            using_native_amp=False,
            using_lbfgs=is_lbfgs
        )

    def clip_gradients(self, optimizer, clip_val=None):
        # apply clip gradients
        # TODO: separate TPU case from here
        self._clip_gradients(optimizer, clip_val)

    def barrier(self, name: Optional[str] = None):
        torch_xla.core.xla_model.rendezvous(f"pl.Trainer.{name}")

    def early_stopping_should_stop(self, pl_module):
        stop = torch.tensor(int(self.trainer.should_stop), device=pl_module.device, dtype=torch.int32)
        stop = xm.mesh_reduce("stop_signal", stop, sum)
        torch_xla.core.xla_model.rendezvous("pl.EarlyStoppingCallback.stop_distributed_training_check")
        should_stop = int(stop.item()) == self.trainer.world_size
        return should_stop

    def save_spawn_weights(self, model):
        """
        Dump a temporary checkpoint after ddp ends to get weights out of the process
        """
        if self.trainer.is_global_zero:
            path = os.path.join(self.trainer.default_root_dir, '__temp_weight_distributed_end.ckpt')
            self.trainer.save_checkpoint(path)
            return path

    def load_spawn_weights(self, original_model):
        """
        Load the temp weights saved in the process
        To recover the trained model from the ddp process we load the saved weights
        """

        loaded_model = original_model

        if self.trainer.is_global_zero:
            # load weights saved in ddp
            path = os.path.join(self.trainer.default_root_dir, '__temp_weight_distributed_end.ckpt')
            loaded_model = original_model.__class__.load_from_checkpoint(path)

            # copy loaded weights to old model
            original_model.load_state_dict(loaded_model.state_dict())

            # remove ddp weights
            os.remove(path)

        return loaded_model

    def transfer_distrib_spawn_state_on_fit_end(self, model, mp_queue, results):
        if self.trainer.distributed_backend not in ("ddp_spawn", "ddp_cpu", "tpu"):
            return

        # track the best model path
        best_model_path = None
        if self.trainer.checkpoint_callback is not None:
            best_model_path = self.trainer.checkpoint_callback.best_model_path

        if self.trainer.global_rank == 0 and mp_queue is not None:
            rank_zero_warn('cleaning up ddp environment...')
            # todo, pass complete checkpoint as state dictionary
            mp_queue.put(best_model_path)
            mp_queue.put(results)

            # save the last weights
            last_path = None
            if not self.trainer.testing and best_model_path is not None and len(best_model_path) > 0:
                last_path = re.sub('.ckpt', '.tmp_end.ckpt', best_model_path)
                atomic_save(model.state_dict(), last_path)
            mp_queue.put(last_path)

    def broadcast(self, obj, src=0):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        data_tensor = torch.tensor(data).to(xm.xla_device(), dtype=torch.float)
        data = xm.all_gather(data_tensor)
        buffer = io.BytesIO(data.cpu().byte().numpy())
        obj = torch.load(buffer)
        return obj

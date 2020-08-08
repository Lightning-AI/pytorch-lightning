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

import os

import torch
import torch.multiprocessing as mp

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class TPUBackend(object):

    def __init__(self, trainer):
        self.trainer = trainer
        self.start_method = None
        self.mp_queue = None

    def setup(self):
        rank_zero_info(f'training on {self.trainer.tpu_cores} TPU cores')

        if not XLA_AVAILABLE:
            raise MisconfigurationException('PyTorch XLA not installed.')

        # see: https://discuss.pytorch.org/t/segfault-with-multiprocessing-queue/81292/2
        self.start_method = 'fork'

        # pass in a state q
        smp = mp.get_context(self.start_method)
        self.mp_queue = smp.SimpleQueue()

    def teardown(self, model):
        # restore main state with best weights
        best_path = self.mp_queue.get()
        results = self.mp_queue.get()
        last_path = self.mp_queue.get()

        # transfer back the best path to the trainer
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

    def train(self, model: LightningModule):
        self.trainer.model = model

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
            self.trainer.load_spawn_weights(model)

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

        # Run the pretrain routine
        results = trainer.run_pretrain_routine(model)

        # save weights at the end of training
        self.__save_end_of_training_weights(model, trainer)

        # persist info in spawn
        trainer.transfer_distrib_spawn_state_on_fit_end(model, mp_queue, results)

    def __save_end_of_training_weights(self, model: LightningModule, trainer):
        # when training ends on these platforms dump weights to get out of the main process
        if trainer.on_colab_kaggle:
            rank_zero_warn('cleaning up... please do not interrupt')
            trainer.save_spawn_weights(model)

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
        optimizers, lr_schedulers, optimizer_frequencies = trainer.init_optimizers(model)
        trainer.optimizers = optimizers
        trainer.lr_schedulers = lr_schedulers
        trainer.optimizer_frequencies = optimizer_frequencies

        # init 16 bit for TPU
        if trainer.precision == 16:
            os.environ['XLA_USE_BF16'] = str(1)

        log.info(f'INIT TPU local core: {trainer.tpu_local_core_rank},'
                 f' global rank: {trainer.tpu_global_core_rank}'
                 f' with XLA_USE_BF16={os.environ.get("XLA_USE_BF16")}')

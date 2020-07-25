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
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning import _logger as log


try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class TPUAccelerator(object):

    def __init__(self, trainer):
        self.trainer = trainer
        self.start_method = None

    def setup(self):
        rank_zero_info(f'training on {self.trainer.tpu_cores} TPU cores')

        if not XLA_AVAILABLE:
            raise MisconfigurationException('No TPU devices found.')

        #  COLAB_GPU is an env var available by default in Colab environments.
        self.start_method = 'fork' if self.trainer.on_colab_kaggle else 'spawn'

    def teardown(self):

        # when training completes, load the weights back in main process
        self.__load_weights_on_main_process()

    def train(self, model):
        self.trainer.model = model

        # train
        if self.trainer.tpu_id is not None:
            self.tpu_train_in_process(self.trainer.tpu_id, model)
        else:
            xmp.spawn(
                self.tpu_train_in_process,
                args=(model,),
                nprocs=self.trainer.tpu_cores,
                start_method=self.start_method
            )

    def __load_weights_on_main_process(self):
        model = self.trainer.model

        # load weights if not interrupted
        if self.trainer.on_colab_kaggle and not self.trainer.testing:
            self.trainer.load_spawn_weights(model)

        self.trainer.model = model

    def tpu_train_in_process(self, tpu_core_idx, model):
        """
        Here we are inside each individual process
        """
        if not self.trainer.trainer.testing:
            self.trainer.setup('fit')
            model.setup('fit')

        # setup TPU training
        self.__setup_tpu_training(model)

        # Run the pretrain routine
        self.trainer.run_pretrain_routine(model)

        # save weights at the end of training
        self.__save_end_of_training_weights(model)

    def __save_end_of_training_weights(self, model):

        # when training ends on these platforms dump weights to get out of the main process
        if self.trainer.on_colab_kaggle:
            rank_zero_warn('cleaning up... please do not interrupt')
            self.trainer.save_spawn_weights(model)

    def __setup_tpu_training(self, model):
        # use the default device from the process
        tpu_device = xm.xla_device()

        # if given an ordinal device, use this as the device
        if self.trainer.tpu_id is not None:
            tpu_device = xm.xla_device(self.trainer.tpu_id)

        # track the device and move model to it
        self.trainer._device = tpu_device
        model.to(self.trainer._device)

        # get the appropriate tpu ranks
        self.trainer.tpu_local_core_rank = xm.get_local_ordinal()
        self.trainer.tpu_global_core_rank = xm.get_ordinal()

        # avoid duplicating progress bar
        if self.trainer.tpu_global_core_rank != 0 and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        self.trainer.global_rank = self.trainer.tpu_local_core_rank
        rank_zero_only.rank = self.trainer.global_rank

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

        # init 16 bit for TPU
        if self.trainer.precision == 16:
            os.environ['XLA_USE_BF16'] = str(1)

        log.info(f'INIT TPU local core: {self.trainer.tpu_local_core_rank},'
                 f' global rank: {self.trainer.tpu_global_core_rank}')

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

from typing import Union

from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class DebuggingConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_init_start(
            self,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            val_check_interval,
            overfit_batches,
            fast_dev_run
    ):
        if not isinstance(fast_dev_run, (bool, int)):
            raise MisconfigurationException(
                f'fast_dev_run={fast_dev_run} is not a valid configuration.'
                ' It should be either a bool or an int >= 0'
            )

        if isinstance(fast_dev_run, int) and (fast_dev_run < 0):
            raise MisconfigurationException(
                f'fast_dev_run={fast_dev_run} is not a'
                ' valid configuration. It should be >= 0.'
            )

        self.trainer.fast_dev_run = fast_dev_run
        fast_dev_run = int(fast_dev_run)

        # set fast_dev_run=True when it is 1, used while logging
        if fast_dev_run == 1:
            self.trainer.fast_dev_run = True

        if fast_dev_run:
            limit_train_batches = fast_dev_run
            limit_val_batches = fast_dev_run
            limit_test_batches = fast_dev_run
            self.trainer.max_steps = fast_dev_run
            self.trainer.num_sanity_val_steps = 0
            self.trainer.max_epochs = 1
            val_check_interval = 1.0
            self.trainer.check_val_every_n_epoch = 1
            self.trainer.logger = DummyLogger()

            rank_zero_info(
                'Running in fast_dev_run mode: will run a full train,'
                f' val and test loop using {fast_dev_run} batch(es).'
            )

        self.trainer.limit_train_batches = _determine_batch_limits(limit_train_batches, 'limit_train_batches')
        self.trainer.limit_val_batches = _determine_batch_limits(limit_val_batches, 'limit_val_batches')
        self.trainer.limit_test_batches = _determine_batch_limits(limit_test_batches, 'limit_test_batches')
        self.trainer.val_check_interval = _determine_batch_limits(val_check_interval, 'val_check_interval')
        self.trainer.overfit_batches = _determine_batch_limits(overfit_batches, 'overfit_batches')
        self.determine_data_use_amount(self.trainer.overfit_batches)

    def determine_data_use_amount(self, overfit_batches: float) -> None:
        """Use less data for debugging purposes"""
        if overfit_batches > 0:
            self.trainer.limit_train_batches = overfit_batches
            self.trainer.limit_val_batches = overfit_batches
            self.trainer.limit_test_batches = overfit_batches


def _determine_batch_limits(batches: Union[int, float], name: str) -> Union[int, float]:
    if 0 <= batches <= 1:
        return batches
    elif batches > 1 and batches % 1.0 == 0:
        return int(batches)
    else:
        raise MisconfigurationException(
            f'You have passed invalid value {batches} for {name}, it has to be in [0.0, 1.0] or an int.'
        )

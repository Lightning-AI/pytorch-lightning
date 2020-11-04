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

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from typing import Union
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_info


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

        self.trainer.fast_dev_run = fast_dev_run
        if self.trainer.fast_dev_run:
            limit_train_batches = 1
            limit_val_batches = 1
            limit_test_batches = 1
            self.trainer.num_sanity_val_steps = 0
            self.trainer.max_epochs = 1
            rank_zero_info(
                'Running in fast_dev_run mode: will run a full train,' ' val and test loop using a single batch'
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

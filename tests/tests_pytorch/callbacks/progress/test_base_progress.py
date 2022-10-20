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
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.trainer.trainer import Trainer


def test_main_progress_bar_with_val_check_interval_int():
    """Test the main progress bar count when val_check_interval=int and check_val_every_n_epoch=None."""
    train_batches = 5
    trainer = Trainer(
        limit_train_batches=train_batches, limit_val_batches=10, val_check_interval=3, check_val_every_n_epoch=None
    )
    model = BoringModel()
    trainer.progress_bar_callback.setup(trainer, model, stage="fit")
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model)
    trainer.reset_train_dataloader()
    trainer.reset_val_dataloader()
    expected = [15, 25, 25, 15]

    for count in expected:
        assert trainer.progress_bar_callback.total_batches_current_epoch == count
        trainer.fit_loop.epoch_loop.batch_progress.total.ready += train_batches

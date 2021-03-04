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

from tests.models.data.horovod.train_default_model import run_test_from_config


def test_horovod_model_script(tmpdir):
    """This just for testing/debugging horovod script without horovod..."""
    trainer_options = dict(
        default_root_dir=str(tmpdir),
        weights_save_path=str(tmpdir),
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        deterministic=True,
    )
    run_test_from_config(trainer_options, check_size=False, on_gpu=False)

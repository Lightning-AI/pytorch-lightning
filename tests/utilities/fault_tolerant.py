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

# Note, this file is used to ensure Fault Tolerant is working as expected
import os

from pytorch_lightning import seed_everything
from tests.utilities.test_auto_restart import _run_training, RandomGetItemDataset

tmpdir = "/tmp/pl_fault_tolerant"

os.makedirs(tmpdir, exist_ok=True)

env_backup = os.environ.copy()

auto_restart_checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
auto_restart_checkpoint_path_exists = os.path.exists(auto_restart_checkpoint_path)

os.environ["PL_FAULT_TOLERANT_TRAINING"] = "1"

seed_everything(42)

dataset_classes = [RandomGetItemDataset]

trainer_kwargs = dict(
    default_root_dir=tmpdir,
    max_epochs=3,
    enable_progress_bar=False,
    enable_model_summary=False,
)

if auto_restart_checkpoint_path_exists:
    fail_on_step = -1
    completed_batches = 5
else:
    fail_on_step = 4
    completed_batches = 4

# Perform a failure
complete_batches, weights = _run_training(trainer_kwargs, dataset_classes, fail_on_step=fail_on_step)
assert len(complete_batches) == completed_batches

if not auto_restart_checkpoint_path_exists:
    checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
    assert os.path.exists(checkpoint_path)
    print(".pl_auto_save.ckpt exists.")

if auto_restart_checkpoint_path_exists:
    print([w for w in weights])

os.environ.clear()
os.environ.update(env_backup)

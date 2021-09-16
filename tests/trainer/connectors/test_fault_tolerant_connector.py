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
import subprocess
import sys
from time import sleep
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@pytest.mark.parametrize("should_gracefully_terminate", [False, True])
@RunIf(min_torch="1.7.0", special=True)
def test_fault_tolerant_sig_handler(should_gracefully_terminate, tmpdir):

    with mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": str(int(should_gracefully_terminate))}):

        class TestModel(BoringModel):
            def training_step(self, batch, batch_idx):
                if should_gracefully_terminate and self.trainer.current_epoch == 1 and batch_idx == 1:
                    env_copy = os.environ.copy()
                    env_copy["PID"] = str(os.getpid())
                    command = [sys.executable, os.path.join(os.path.dirname(__file__), "fault_tolerant_pid_killer.py")]
                    subprocess.Popen(command, env=env_copy)
                    sleep(0.1)
                return super().training_step(batch, batch_idx)

        model = TestModel()
        trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_train_batches=2, limit_val_batches=2)
        trainer.fit(model)
        assert trainer._should_gracefully_terminate == should_gracefully_terminate

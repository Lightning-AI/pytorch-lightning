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
import glob
import os
import sys
from pathlib import Path

import pytest

from pytorch_lightning import Callback, Trainer
from tests import _PATH_LEGACY
from tests.helpers import BoringModel

LEGACY_CHECKPOINTS_PATH = os.path.join(_PATH_LEGACY, "checkpoints")
CHECKPOINT_EXTENSION = ".ckpt"


# todo: add more legacy checkpoints - for < v0.8
@pytest.mark.parametrize(
    "pl_version",
    [
        # "0.8.1",
        "0.8.3",
        "0.8.4",
        # "0.8.5", # this version has problem with loading on PT<=1.4 as it seems to be archive
        # "0.9.0", # this version has problem with loading on PT<=1.4 as it seems to be archive
        "0.10.0",
        "1.0.0",
        "1.0.1",
        "1.0.2",
        "1.0.3",
        "1.0.4",
        "1.0.5",
        "1.0.6",
        "1.0.7",
        "1.0.8",
        "1.1.0",
        "1.1.1",
        "1.1.2",
        "1.1.3",
        "1.1.4",
        "1.1.5",
        "1.1.6",
        "1.1.7",
        "1.1.8",
        "1.2.0",
        "1.2.1",
        "1.2.2",
        "1.2.3",
        "1.2.4",
        "1.2.5",
        "1.2.6",
        "1.2.7",
        "1.2.8",
        "1.2.10",
        "1.3.0",
        "1.3.1",
        "1.3.2",
        "1.3.3",
        "1.3.4",
        "1.3.5",
        "1.3.6",
        "1.3.7",
        "1.3.8",
    ],
)
def test_resume_legacy_checkpoints(tmpdir, pl_version: str):
    path_dir = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)

    # todo: make this as mock, so it is cleaner...
    orig_sys_paths = list(sys.path)
    sys.path.insert(0, path_dir)
    from zero_training import DummyModel

    path_ckpts = sorted(glob.glob(os.path.join(path_dir, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, 'No checkpoints found in folder "%s"' % path_dir
    path_ckpt = path_ckpts[-1]

    model = DummyModel.load_from_checkpoint(path_ckpt)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=6)
    trainer.fit(model)

    # todo
    # model = DummyModel()
    # trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, resume_from_checkpoint=path_ckpt)
    # trainer.fit(model)

    sys.path = orig_sys_paths


class OldStatefulCallback(Callback):

    def __init__(self, state):
        self.state = state

    @property
    def state_id(self):
        return type(self)

    def on_save_checkpoint(self, *args):
        return {"state": self.state}

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state = callback_state["state"]


def test_resume_callback_state_saved_by_type(tmpdir):
    """ Test that a legacy checkpoint that didn't use a state identifier before can still be loaded. """
    model = BoringModel()
    callback = OldStatefulCallback(state=111)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
        callbacks=[callback],
    )
    trainer.fit(model)
    ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    assert ckpt_path.exists()

    callback = OldStatefulCallback(state=222)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=2,
        callbacks=[callback],
        resume_from_checkpoint=ckpt_path,
    )
    trainer.fit(model)
    assert callback.state == 111

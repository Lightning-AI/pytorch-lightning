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
from unittest.mock import patch

import pytest

from pytorch_lightning import Trainer
from tests import _PATH_LEGACY

LEGACY_CHECKPOINTS_PATH = os.path.join(_PATH_LEGACY, 'checkpoints')
CHECKPOINT_EXTENSION = ".ckpt"
LEGACY_BACK_COMPATIBLE_PL_VERSIONS = (
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
)


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
def test_load_legacy_checkpoints(tmpdir, pl_version: str):
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    with patch('sys.path', [PATH_LEGACY] + sys.path):
        from simple_classif_training import ClassifDataModule, ClassificationModel

        path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f'*{CHECKPOINT_EXTENSION}')))
        assert path_ckpts, 'No checkpoints found in folder "%s"' % PATH_LEGACY
        path_ckpt = path_ckpts[-1]

        model = ClassificationModel.load_from_checkpoint(path_ckpt)
        trainer = Trainer(default_root_dir=str(tmpdir))
        dm = ClassifDataModule()
        res = trainer.test(model, dm)
        assert res[0]['test_loss'] <= 0.7
        assert res[0]['test_acc'] >= 0.85
        print(res)


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
def test_resume_legacy_checkpoints(tmpdir, pl_version: str):
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    with patch('sys.path', [PATH_LEGACY] + sys.path):
        from simple_classif_training import ClassifDataModule, ClassificationModel

        path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f'*{CHECKPOINT_EXTENSION}')))
        assert path_ckpts, 'No checkpoints found in folder "%s"' % PATH_LEGACY
        path_ckpt = path_ckpts[-1]

        dm = ClassifDataModule()
        model = ClassificationModel()
        trainer = Trainer(default_root_dir=str(tmpdir), max_epochs=16, resume_from_checkpoint=path_ckpt)
        trainer.fit(model, dm)
        res = trainer.test(model, dm)
        assert res[0]['test_loss'] <= 0.7
        assert res[0]['test_acc'] >= 0.85
        print(res)

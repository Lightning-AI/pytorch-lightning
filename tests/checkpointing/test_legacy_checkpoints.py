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
import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from tests import _PATH_LEGACY

LEGACY_CHECKPOINTS_PATH = os.path.join(_PATH_LEGACY, "checkpoints")
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
    with patch("sys.path", [PATH_LEGACY] + sys.path):
        from simple_classif_training import ClassifDataModule, ClassificationModel

        path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f"*{CHECKPOINT_EXTENSION}")))
        assert path_ckpts, f'No checkpoints found in folder "{PATH_LEGACY}"'
        path_ckpt = path_ckpts[-1]

        model = ClassificationModel.load_from_checkpoint(path_ckpt)
        trainer = Trainer(default_root_dir=str(tmpdir))
        dm = ClassifDataModule()
        res = trainer.test(model, datamodule=dm)
        assert res[0]["test_loss"] <= 0.7
        assert res[0]["test_acc"] >= 0.85
        print(res)


class LimitNbEpochs(Callback):
    def __init__(self, nb: int):
        self.limit = nb
        self._count = 0

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._count += 1
        if self._count >= self.limit:
            trainer.should_stop = True


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
def test_resume_legacy_checkpoints(tmpdir, pl_version: str):
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    with patch("sys.path", [PATH_LEGACY] + sys.path):
        from simple_classif_training import ClassifDataModule, ClassificationModel

        path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f"*{CHECKPOINT_EXTENSION}")))
        assert path_ckpts, f'No checkpoints found in folder "{PATH_LEGACY}"'
        path_ckpt = path_ckpts[-1]

        dm = ClassifDataModule()
        model = ClassificationModel()
        es = EarlyStopping(monitor="val_acc", mode="max", min_delta=0.005)
        stop = LimitNbEpochs(1)
        trainer = Trainer(
            default_root_dir=str(tmpdir),
            gpus=int(torch.cuda.is_available()),
            precision=(16 if torch.cuda.is_available() else 32),
            checkpoint_callback=True,
            callbacks=[es, stop],
            max_epochs=21,
            accumulate_grad_batches=2,
            deterministic=True,
            resume_from_checkpoint=path_ckpt,
        )
        trainer.fit(model, datamodule=dm)
        res = trainer.test(model, datamodule=dm)
        assert res[0]["test_loss"] <= 0.7
        assert res[0]["test_acc"] >= 0.85
        print(res)

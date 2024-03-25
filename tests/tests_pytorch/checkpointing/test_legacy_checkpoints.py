# Copyright The Lightning AI team.
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

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch import Callback, Trainer

from tests_pytorch import _PATH_LEGACY
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel
from tests_pytorch.helpers.threading import ThreadExceptionHandler

LEGACY_CHECKPOINTS_PATH = os.path.join(_PATH_LEGACY, "checkpoints")
CHECKPOINT_EXTENSION = ".ckpt"
# load list of all back compatible versions
with open(os.path.join(_PATH_LEGACY, "back-compatible-versions.txt")) as fp:
    LEGACY_BACK_COMPATIBLE_PL_VERSIONS = [ln.strip() for ln in fp.readlines()]
# This shall be created for each CI run
LEGACY_BACK_COMPATIBLE_PL_VERSIONS += ["local"]


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@RunIf(sklearn=True)
def test_load_legacy_checkpoints(tmp_path, pl_version: str):
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    with patch("sys.path", [PATH_LEGACY] + sys.path):
        path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f"*{CHECKPOINT_EXTENSION}")))
        assert path_ckpts, f'No checkpoints found in folder "{PATH_LEGACY}"'
        path_ckpt = path_ckpts[-1]

        model = ClassificationModel.load_from_checkpoint(path_ckpt, num_features=24)
        trainer = Trainer(default_root_dir=tmp_path)
        dm = ClassifDataModule(num_features=24, length=6000, batch_size=128, n_clusters_per_class=2, n_informative=8)
        res = trainer.test(model, datamodule=dm)
        assert res[0]["test_loss"] <= 0.85, str(res[0]["test_loss"])
        assert res[0]["test_acc"] >= 0.7, str(res[0]["test_acc"])
        print(res)


class LimitNbEpochs(Callback):
    def __init__(self, nb: int):
        self.limit = nb
        self._count = 0

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._count += 1
        if self._count >= self.limit:
            trainer.should_stop = True


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@RunIf(sklearn=True)
def test_legacy_ckpt_threading(pl_version: str):
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, f'No checkpoints found in folder "{PATH_LEGACY}"'
    path_ckpt = path_ckpts[-1]

    def load_model():
        import torch
        from lightning.pytorch.utilities.migration import pl_legacy_patch

        with pl_legacy_patch():
            _ = torch.load(path_ckpt)

    with patch("sys.path", [PATH_LEGACY] + sys.path):
        t1 = ThreadExceptionHandler(target=load_model)
        t2 = ThreadExceptionHandler(target=load_model)

        t1.start()
        t2.start()

        t1.join()
        t2.join()


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@RunIf(sklearn=True)
def test_resume_legacy_checkpoints(tmp_path, pl_version: str):
    PATH_LEGACY = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    with patch("sys.path", [PATH_LEGACY] + sys.path):
        path_ckpts = sorted(glob.glob(os.path.join(PATH_LEGACY, f"*{CHECKPOINT_EXTENSION}")))
        assert path_ckpts, f'No checkpoints found in folder "{PATH_LEGACY}"'
        path_ckpt = path_ckpts[-1]

        dm = ClassifDataModule(num_features=24, length=6000, batch_size=128, n_clusters_per_class=2, n_informative=8)
        model = ClassificationModel(num_features=24)
        stop = LimitNbEpochs(1)

        trainer = Trainer(
            default_root_dir=tmp_path,
            accelerator="auto",
            devices=1,
            precision=("16-mixed" if torch.cuda.is_available() else "32-true"),
            callbacks=[stop],
            max_epochs=21,
            accumulate_grad_batches=2,
        )
        torch.backends.cudnn.deterministic = True
        trainer.fit(model, datamodule=dm, ckpt_path=path_ckpt)
        res = trainer.test(model, datamodule=dm)
        assert res[0]["test_loss"] <= 0.85, str(res[0]["test_loss"])
        assert res[0]["test_acc"] >= 0.7, str(res[0]["test_acc"])

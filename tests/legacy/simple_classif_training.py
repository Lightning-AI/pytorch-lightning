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
import os
import sys

import lightning.pytorch as pl
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.simple_models import ClassificationModel

PATH_LEGACY = os.path.dirname(__file__)


def main_train(dir_path, max_epochs: int = 20):
    seed_everything(42)
    stopping = EarlyStopping(monitor="val_acc", mode="max", min_delta=0.005)
    trainer = pl.Trainer(
        accelerator="auto",
        default_root_dir=dir_path,
        precision=(16 if torch.cuda.is_available() else 32),
        callbacks=[stopping],
        min_epochs=3,
        max_epochs=max_epochs,
        accumulate_grad_batches=2,
        deterministic=True,
    )

    dm = ClassifDataModule(
        num_features=24, length=6000, num_classes=3, batch_size=128, n_clusters_per_class=2, n_informative=int(24 / 3)
    )
    model = ClassificationModel(num_features=24, num_classes=3, lr=0.01)
    trainer.fit(model, datamodule=dm)
    res = trainer.test(model, datamodule=dm)
    assert res[0]["test_loss"] <= 0.85, str(res[0]["test_loss"])
    assert res[0]["test_acc"] >= 0.7, str(res[0]["test_acc"])
    assert trainer.current_epoch < (max_epochs - 1)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else str(pl.__version__)
    path_dir = os.path.join(PATH_LEGACY, "checkpoints", name)
    main_train(path_dir)

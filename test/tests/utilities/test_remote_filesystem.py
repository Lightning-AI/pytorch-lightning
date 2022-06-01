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

import fsspec
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tests.helpers import BoringModel

GCS_BUCKET_PATH = os.getenv("GCS_BUCKET_PATH", None)
_GCS_BUCKET_PATH_AVAILABLE = GCS_BUCKET_PATH is not None

gcs_fs = fsspec.filesystem("gs") if _GCS_BUCKET_PATH_AVAILABLE else None


def gcs_path_join(dir_path):
    return GCS_BUCKET_PATH + str(dir_path)


def gcs_rm_dir(dir_path):
    gcs_fs.rm(dir_path, recursive=True)
    return True


@pytest.mark.skipif(not _GCS_BUCKET_PATH_AVAILABLE, reason="Test requires GCS bucket path")
def test_gcs_model_checkpoint_contents(tmpdir):
    dir_path = gcs_path_join(tmpdir)

    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=dir_path, save_top_k=-1, save_last=True)
    epochs = 2

    trainer = Trainer(
        default_root_dir=dir_path,
        callbacks=[checkpoint_callback],
        limit_train_batches=10,
        limit_val_batches=10,
        max_epochs=2,
        logger=False,
    )

    trainer.fit(model)

    assert checkpoint_callback.best_model_path == os.path.join(dir_path, "epoch=1-step=19.ckpt")
    assert checkpoint_callback.last_model_path == os.path.join(dir_path, "last.ckpt")

    expected = [f"epoch={i}-step={j}.ckpt" for i, j in zip(range(epochs), [9, 19])]
    expected.append("last.ckpt")

    gcs_ckpt_paths = [os.path.basename(path) for path in gcs_fs.listdir(dir_path, detail=False)]
    assert gcs_ckpt_paths == expected

    assert gcs_rm_dir(dir_path)


@pytest.mark.skipif(not _GCS_BUCKET_PATH_AVAILABLE, reason="Test requires GCS bucket path")
def test_gcs_logging(tmpdir):
    dir_path = gcs_path_join(tmpdir)

    name = "tb_versioning"
    log_dir = os.path.join(dir_path, name)
    gcs_fs.mkdir(log_dir)
    expected_version = "101"

    logger = TensorBoardLogger(save_dir=dir_path, name=name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})

    assert logger.version == expected_version

    gcs_paths = [os.path.basename(path) for path in gcs_fs.listdir(log_dir, detail=False)]
    gcs_paths = list(filter(lambda x: len(x) > 0, gcs_paths))

    assert gcs_paths == [expected_version]
    assert gcs_fs.listdir(os.path.join(log_dir, expected_version), detail=False)

    assert gcs_rm_dir(dir_path)


@pytest.mark.skipif(not _GCS_BUCKET_PATH_AVAILABLE, reason="Test requires GCS bucket path")
def test_gcs_save_hparams_to_yaml_file(tmpdir):
    dir_path = gcs_path_join(tmpdir)

    model = BoringModel()
    logger = TensorBoardLogger(save_dir=dir_path, default_hp_metric=False)
    trainer = Trainer(max_steps=1, default_root_dir=dir_path, logger=logger)
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model)

    hparams_file = "hparams.yaml"
    assert gcs_fs.isfile(os.path.join(trainer.log_dir, hparams_file))

    assert gcs_rm_dir(dir_path)

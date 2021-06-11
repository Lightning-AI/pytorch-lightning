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
from pytorch_lightning.loggers import TensorBoardLogger
from tests.helpers import BoringModel

GCS_BUCKET_PATH = os.getenv("GCS_BUCKET_PATH", None)
_GCS_BUCKET_PATH_AVAILABLE = GCS_BUCKET_PATH is not None

gcs_fs = fsspec.filesystem("gs") if _GCS_BUCKET_PATH_AVAILABLE else None


def gcs_path_join(dir_path):
    return GCS_BUCKET_PATH + str(dir_path)


@pytest.mark.skipif(not _GCS_BUCKET_PATH_AVAILABLE, reason="Test requires GCS bucket patch")
def test_gcs_save_hparams_to_yaml_file(tmpdir):
    dir_path = gcs_path_join(tmpdir)

    model = BoringModel()
    logger = TensorBoardLogger(save_dir=dir_path, default_hp_metric=False)
    trainer = Trainer(max_steps=1, default_root_dir=dir_path, logger=logger)
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model)

    hparams_file = "hparams.yaml"
    assert gcs_fs.isfile(os.path.join(trainer.log_dir, hparams_file))

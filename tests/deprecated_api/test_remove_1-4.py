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
"""Test deprecated functionality which will be removed in v1.4.0"""

import pytest

from pytorch_lightning import Trainer
from tests.deprecated_api import _soft_unimport_module
from tests.helpers import BoringModel


def test_v1_4_0_deprecated_imports():
    _soft_unimport_module('pytorch_lightning.utilities.argparse_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.argparse_utils import _gpus_arg_default  # noqa: F811 F401


def test_v1_4_0_deprecated_hpc_load(tmpdir):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
    )
    trainer.fit(model)
    trainer.checkpoint_connector.hpc_save(tmpdir, trainer.logger)
    checkpoint_path = trainer.checkpoint_connector.get_max_ckpt_path_from_folder(str(tmpdir))
    with pytest.deprecated_call(match=r"`CheckpointConnector.hpc_load\(\)` was deprecated in v1.4"):
        trainer.checkpoint_connector.hpc_load(checkpoint_path)

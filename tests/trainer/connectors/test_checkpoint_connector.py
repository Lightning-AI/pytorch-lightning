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
from pathlib import Path

import torch

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


def test_preloaded_checkpoint_lifecycle(tmpdir):
    """ Tests that the preloaded checkpoint contents gets cleared from memory when it is not required anymore. """
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
    )
    trainer.fit(model)

    connector = trainer.checkpoint_connector

    assert not trainer.resume_from_checkpoint
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint

    connector.resume_start()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint
    connector.resume_end()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint

    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, resume_from_checkpoint=ckpt_path)
    connector = trainer.checkpoint_connector
    connector.resume_start()
    assert connector.resume_checkpoint_path == ckpt_path
    assert connector._loaded_checkpoint
    assert isinstance(connector._loaded_checkpoint, dict)
    connector.resume_end()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint


def test_hpc_restore_attempt(tmpdir):
    """ Test that restore() attempts to restore the hpc_ckpt with highest priority. """
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
    )
    trainer.fit(model)

    hpc_ckpt_path = tmpdir / "hpc_ckpt_3.ckpt"
    trainer.save_checkpoint(hpc_ckpt_path)
    assert Path(hpc_ckpt_path).exists()

    # set weights to zero
    for param in model.parameters():
        torch.nn.init.constant_(param, 0)

    # case 1: restore hpc first, no explicit resume path provided
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=2,
    )
    trainer.fit(model)

    for param in model.parameters():
        assert param.abs().sum() > 0
        torch.nn.init.constant_(param, 0)

    # case 2: explicit resume path provided, restore hpc anyway
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, resume_from_checkpoint="not existing")
    trainer.fit(model)

    for param in model.parameters():
        assert param.abs().sum() > 0


def test_hpc_max_ckpt_version(tmpdir):
    """ Test that the CheckpointConnector is able to find the hpc checkpoint file with the highest version. """
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
    )
    trainer.fit(model)
    trainer.save_checkpoint(tmpdir / "hpc_ckpt.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_0.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_3.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_33.ckpt")

    assert trainer.checkpoint_connector.hpc_resume_path == tmpdir / "hpc_ckpt_33.ckpt"
    assert trainer.checkpoint_connector.max_ckpt_version_in_folder(tmpdir) == 33
    assert trainer.checkpoint_connector.max_ckpt_version_in_folder(tmpdir / "not" / "existing") is None
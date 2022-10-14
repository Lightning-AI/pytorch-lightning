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
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.trainer.states import TrainerFn


def test_preloaded_checkpoint_lifecycle(tmpdir):
    """Tests that the preloaded checkpoint contents gets cleared from memory when it is not required anymore."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)

    connector = trainer._checkpoint_connector

    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint

    connector.resume_start()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint
    connector.resume_end()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint

    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2)
    connector = trainer._checkpoint_connector
    connector.resume_start(ckpt_path)
    assert connector.resume_checkpoint_path == ckpt_path
    assert connector._loaded_checkpoint
    assert isinstance(connector._loaded_checkpoint, dict)
    trainer.state.fn = TrainerFn.FITTING
    connector.resume_end()
    assert not connector.resume_checkpoint_path
    assert not connector._loaded_checkpoint


@mock.patch("lightning_lite.plugins.environments.slurm_environment.SLURMEnvironment.detect", return_value=True)
def test_hpc_restore_attempt(_, tmpdir):
    """Test that restore() attempts to restore the hpc_ckpt with highest priority."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, enable_checkpointing=False, logger=False)
    trainer.fit(model)

    hpc_ckpt_path = tmpdir / "hpc_ckpt_3.ckpt"
    trainer.save_checkpoint(hpc_ckpt_path)
    assert os.listdir(tmpdir) == ["hpc_ckpt_3.ckpt"]

    # set weights to zero
    for param in model.parameters():
        torch.nn.init.constant_(param, 0)

    # case 1: restore hpc first, no explicit resume path provided
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, enable_checkpointing=False, logger=False)
    trainer.fit(model)

    for param in model.parameters():
        assert param.abs().sum() > 0
        torch.nn.init.constant_(param, 0)

    # case 2: explicit resume path provided, file not found
    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)

    with pytest.raises(FileNotFoundError, match="Checkpoint at not existing not found. Aborting training."):
        trainer.fit(model, ckpt_path="not existing")


def test_hpc_max_ckpt_version(tmpdir):
    """Test that the CheckpointConnector is able to find the hpc checkpoint file with the highest version."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)
    trainer.save_checkpoint(tmpdir / "hpc_ckpt.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_0.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_3.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_33.ckpt")

    assert trainer._checkpoint_connector._hpc_resume_path == str(tmpdir / "hpc_ckpt_33.ckpt")
    assert trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmpdir) == 33
    assert (
        trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmpdir / "not" / "existing")
        is None
    )


def test_loops_restore(tmpdir):
    """Test that required loop state_dict is loaded correctly by checkpoint connector."""
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    trainer_args = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )
    trainer = Trainer(**trainer_args)
    trainer.fit(model)

    ckpt_path = str(tmpdir / "last.ckpt")

    trainer = Trainer(**trainer_args)
    trainer.strategy.connect(model)

    trainer_fns = [fn for fn in TrainerFn._without_tune()]

    for fn in trainer_fns:
        trainer_fn = getattr(trainer, f"{fn}_loop")
        trainer_fn.load_state_dict = mock.Mock()

    for fn in trainer_fns:
        trainer.state.fn = fn
        trainer._checkpoint_connector.resume_start(ckpt_path)
        trainer._checkpoint_connector.restore_loops()

        trainer_loop = getattr(trainer, f"{fn}_loop")
        trainer_loop.load_state_dict.assert_called()
        trainer_loop.load_state_dict.reset_mock()

        for fn2 in trainer_fns:
            if fn2 != fn:
                trainer_loop2 = getattr(trainer, f"{fn2}_loop")
                trainer_loop2.load_state_dict.assert_not_called()

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
from typing import Mapping
from unittest import mock
from unittest.mock import Mock

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerFn
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class HPCHookdedModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.hpc_save_called = 0
        self.hpc_load_called = 0

    def on_hpc_save(self, checkpoint):
        assert "state_dict" in checkpoint
        self.hpc_save_called += 1

    def on_hpc_load(self, checkpoint):
        assert "state_dict" in checkpoint
        self.hpc_load_called += 1


def test_hpc_hook_calls(tmpdir):
    model = HPCHookdedModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, checkpoint_callback=False, logger=False)
    trainer.fit(model)
    connector = trainer.checkpoint_connector
    connector.hpc_save(tmpdir, logger=Mock())
    assert model.hpc_save_called == 1
    assert model.hpc_load_called == 0

    # new training run, restore from hpc checkpoint file automatically
    assert set(os.listdir(tmpdir)) == {"hpc_ckpt_1.ckpt"}
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, checkpoint_callback=False, logger=False)
    trainer.fit(model)
    assert model.hpc_save_called == 1
    assert model.hpc_load_called == 1


def test_preloaded_checkpoint_lifecycle(tmpdir):
    """Tests that the preloaded checkpoint contents gets cleared from memory when it is not required anymore."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
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
    """Test that restore() attempts to restore the hpc_ckpt with highest priority."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, checkpoint_callback=False, logger=False)
    trainer.fit(model)

    hpc_ckpt_path = tmpdir / "hpc_ckpt_3.ckpt"
    trainer.save_checkpoint(hpc_ckpt_path)
    assert os.listdir(tmpdir) == ["hpc_ckpt_3.ckpt"]

    # set weights to zero
    for param in model.parameters():
        torch.nn.init.constant_(param, 0)

    # case 1: restore hpc first, no explicit resume path provided
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, checkpoint_callback=False, logger=False)
    trainer.fit(model)

    for param in model.parameters():
        assert param.abs().sum() > 0
        torch.nn.init.constant_(param, 0)

    # case 2: explicit resume path provided, restore hpc anyway
    trainer = Trainer(default_root_dir=tmpdir, max_steps=3, resume_from_checkpoint="not existing")
    trainer.fit(model)

    for param in model.parameters():
        assert param.abs().sum() > 0


def test_hpc_max_ckpt_version(tmpdir):
    """Test that the CheckpointConnector is able to find the hpc checkpoint file with the highest version."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)
    trainer.save_checkpoint(tmpdir / "hpc_ckpt.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_0.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_3.ckpt")
    trainer.save_checkpoint(tmpdir / "hpc_ckpt_33.ckpt")

    assert trainer.checkpoint_connector.hpc_resume_path == str(tmpdir / "hpc_ckpt_33.ckpt")
    assert trainer.checkpoint_connector.max_ckpt_version_in_folder(tmpdir) == 33
    assert trainer.checkpoint_connector.max_ckpt_version_in_folder(tmpdir / "not" / "existing") is None


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(min_torch="1.7.0")
def test_loops_restore(tmpdir):
    """Test that required loop state_dict is loaded correctly by checkpoint connector."""
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    trainer_args = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=False,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )
    trainer = Trainer(**trainer_args)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)
    trainer.fit(model)

    resume_ckpt = str(tmpdir / "last.ckpt")
    state_dict = torch.load(resume_ckpt)
    trainer_args.update(
        {"max_epochs": 3, "resume_from_checkpoint": resume_ckpt, "checkpoint_callback": False, "callbacks": []}
    )

    def _is_equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.all(torch.eq(a, b))

        if isinstance(a, Mapping):
            return all(_is_equal(a.get(k, None), b.get(k, None)) for k in b.keys() if k != "device")

        return a == b

    for fn in TrainerFn:
        if fn == TrainerFn.TUNING:
            continue

        trainer = Trainer(**trainer_args)
        trainer.state.fn = fn
        trainer.checkpoint_connector.resume_start()
        trainer.checkpoint_connector.restore_loops()
        fn_loop = f"{fn}_loop"
        assert _is_equal(state_dict["loops"][fn_loop], getattr(trainer, fn_loop).state_dict())

        for fn2 in TrainerFn:
            if fn2 != fn and fn2 != TrainerFn.TUNING:
                fn2_loop = f"{fn2}_loop"
                assert not _is_equal(state_dict["loops"][fn2_loop], getattr(trainer, fn2_loop).state_dict())

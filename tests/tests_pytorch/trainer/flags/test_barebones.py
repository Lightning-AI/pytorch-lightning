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
import logging

import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.profilers import PassThroughProfiler


def test_barebones_disables_logging():
    pl_module = BoringModel()
    trainer = Trainer(barebones=True)
    pl_module._trainer = trainer

    with pytest.warns(match=r"barebones=True\)` is configured"):
        pl_module.log("foo", 1.0)
    with pytest.warns(match=r"barebones=True\)` is configured"):
        pl_module.log_dict({"foo": 1.0})


def test_barebones_argument_selection(caplog):
    with caplog.at_level(logging.INFO):
        trainer = Trainer(barebones=True)

    assert "running in `Trainer(barebones=True)` mode" in caplog.text
    assert trainer.barebones
    assert not trainer.checkpoint_callbacks
    assert not trainer.loggers
    assert not trainer.progress_bar_callback
    assert not any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)
    assert not trainer.log_every_n_steps
    assert not trainer.num_sanity_val_steps
    assert not trainer.fast_dev_run
    assert not trainer._detect_anomaly
    assert isinstance(trainer.profiler, PassThroughProfiler)


def test_barebones_raises():
    with pytest.raises(ValueError, match=r"enable_checkpointing=True\)` was passed"):
        Trainer(barebones=True, enable_checkpointing=True)
    with pytest.raises(ValueError, match=r"logger=True\)` was passed"):
        Trainer(barebones=True, logger=True)
    with pytest.raises(ValueError, match=r"enable_progress_bar=True\)` was passed"):
        Trainer(barebones=True, enable_progress_bar=True)
    with pytest.raises(ValueError, match=r"enable_model_summary=True\)` was passed"):
        Trainer(barebones=True, enable_model_summary=True)
    with pytest.raises(ValueError, match=r"log_every_n_steps=1\)` was passed"):
        Trainer(barebones=True, log_every_n_steps=1)
    with pytest.raises(ValueError, match=r"num_sanity_val_steps=1\)` was passed"):
        Trainer(barebones=True, num_sanity_val_steps=1)
    with pytest.raises(ValueError, match=r"fast_dev_run=1\)` was passed"):
        Trainer(barebones=True, fast_dev_run=1)
    with pytest.raises(ValueError, match=r"detect_anomaly=True\)` was passed"):
        Trainer(barebones=True, detect_anomaly=True)
    with pytest.raises(ValueError, match=r"profiler='simple'\)` was passed"):
        Trainer(barebones=True, profiler="simple")

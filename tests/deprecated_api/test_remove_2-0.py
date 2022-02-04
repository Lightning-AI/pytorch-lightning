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
"""Test deprecated functionality which will be removed in v2.0.0."""
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from tests.helpers.runif import RunIf
from tests.callbacks.test_callbacks import OldStatefulCallback
from tests.helpers import BoringModel

def test_v2_0_0_deprecated_num_processes(tmpdir):
    with pytest.deprecated_call(match=r"is deprecated in v1.6 and will be removed in v2.0."):
        _ = Trainer(default_root_dir=tmpdir, num_processes=2)


@mock.patch("torch.cuda.is_available", return_value=True)
def test_v2_0_0_deprecated_gpus(tmpdir):
    with pytest.deprecated_call(match=r"is deprecated in v1.6 and will be removed in v2.0."):
        _ = Trainer(default_root_dir=tmpdir, gpus=2)


def test_v2_0_0_deprecated_tpu_cores(tmpdir):
    with pytest.deprecated_call(match=r"is deprecated in v1.6 and will be removed in v2.0."):
        _ = Trainer(default_root_dir=tmpdir, tpu_cores=1)


@RunIf(ipu=True)
def test_v2_0_0_deprecated_ipus(tmpdir):
    with pytest.deprecated_call(match=r"is deprecated in v1.6 and will be removed in v2.0."):
        _ = Trainer(default_root_dir=tmpdir, ipus=2)


def test_v2_0_resume_from_checkpoint_trainer_constructor(tmpdir):
    # test resume_from_checkpoint still works until v2.0 deprecation
    model = BoringModel()
    callback = OldStatefulCallback(state=111)
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, callbacks=[callback])
    trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path

    callback = OldStatefulCallback(state=222)
    with pytest.deprecated_call(match=r"Setting `Trainer\(resume_from_checkpoint=\)` is deprecated in v1.5"):
        trainer = Trainer(default_root_dir=tmpdir, max_steps=2, callbacks=[callback], resume_from_checkpoint=ckpt_path)
    with pytest.deprecated_call(match=r"trainer.resume_from_checkpoint` is deprecated in v1.5"):
        _ = trainer.resume_from_checkpoint
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path == ckpt_path
    trainer.validate(model=model, ckpt_path=ckpt_path)
    assert callback.state == 222
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path == ckpt_path
    with pytest.deprecated_call(match=r"trainer.resume_from_checkpoint` is deprecated in v1.5"):
        trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path  # last `fit` replaced the `best_model_path`
    assert callback.state == 111
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path is None
    trainer.predict(model=model, ckpt_path=ckpt_path)
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path is None
    trainer.fit(model)
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path is None

    # test fit(ckpt_path=) precedence over Trainer(resume_from_checkpoint=) path
    model = BoringModel()
    with pytest.deprecated_call(match=r"Setting `Trainer\(resume_from_checkpoint=\)` is deprecated in v1.5"):
        trainer = Trainer(resume_from_checkpoint="trainer_arg_path")
    with pytest.raises(FileNotFoundError, match="Checkpoint at fit_arg_ckpt_path not found. Aborting training."):
        trainer.fit(model, ckpt_path="fit_arg_ckpt_path")

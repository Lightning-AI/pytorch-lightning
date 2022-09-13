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
import pytest
from lightning_utilities.core.imports import RequirementCache
from torch import nn

from pytorch_lightning import Trainer
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.utilities.meta import _is_deferred
from tests_pytorch.helpers.runif import RunIf

_TORCHDISTX_AVAILABLE = RequirementCache("torchdistx")


class SimpleBoringModel(LightningModule):
    def __init__(self, num_layers):
        super().__init__()
        self.layer = nn.Sequential(*[nn.Linear(1, 1) for _ in range(num_layers)])


@pytest.mark.skipif(not _TORCHDISTX_AVAILABLE, reason=_TORCHDISTX_AVAILABLE.message)
def test_deferred_init_with_lightning_module():
    from torchdistx.deferred_init import deferred_init, materialize_module
    from torchdistx.fake import is_fake

    model = deferred_init(SimpleBoringModel, 4)
    weight = model.layer[0].weight
    assert weight.device.type == "cpu"
    assert is_fake(weight)
    assert _is_deferred(model)

    materialize_module(model)
    materialize_module(model)  # make sure it's idempotent
    assert not _is_deferred(model)
    weight = model.layer[0].weight
    assert weight.device.type == "cpu"
    assert not is_fake(weight)


@pytest.mark.skipif(not _TORCHDISTX_AVAILABLE, reason=_TORCHDISTX_AVAILABLE.message)
@pytest.mark.parametrize(
    "trainer_kwargs",
    (
        {"accelerator": "auto", "devices": 1},
        pytest.param(
            {"strategy": "deepspeed_stage_3", "accelerator": "gpu", "devices": 2, "precision": 16},
            marks=RunIf(min_cuda_gpus=2, deepspeed=True),
        ),
    ),
)
def test_deferred_init_with_trainer(tmpdir, trainer_kwargs):
    from torchdistx.deferred_init import deferred_init

    model = deferred_init(BoringModel)
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        **trainer_kwargs
    )
    trainer.fit(model)


@pytest.mark.skipif(not _TORCHDISTX_AVAILABLE, reason=_TORCHDISTX_AVAILABLE.message)
def test_deferred_init_ddp_spawn(tmpdir):
    from torchdistx.deferred_init import deferred_init

    model = deferred_init(BoringModel)
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="auto",
        devices="1",
        strategy="ddp_spawn",
    )
    with pytest.raises(NotImplementedError, match="DDPSpawnStrategy` strategy does not support.*torchdistx"):
        trainer.fit(model)

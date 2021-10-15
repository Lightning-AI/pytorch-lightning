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
from torch import nn

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.imports import _TORCH_META_AVAILABLE
from pytorch_lightning.utilities.meta import init_meta, init_meta_context, materialize_module


class MLP(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layer = nn.Sequential(*[nn.Linear(1, 1) for _ in range(num_layers)] + [nn.Dropout(), nn.LayerNorm(1)])


class BoringModel(LightningModule):
    def __init__(self, num_layers: int):
        super().__init__()
        self.save_hyperparameters()
        self.layer = nn.Sequential(*[nn.Linear(1, 1) for _ in range(self.hparams.num_layers)])


@pytest.mark.skipif(not _TORCH_META_AVAILABLE, reason="Support only with PyTorch 1.10")
def test_init_meta_context():

    init_meta(MLP, 4).materialize()

    with init_meta_context():
        m = nn.Linear(in_features=1, out_features=1)
        assert m.weight.device.type == "meta"
        mlp = MLP(4)
        assert mlp.layer[0].weight.device.type == "meta"

        materialize_module(mlp)
        assert mlp.layer[0].weight.device.type == "cpu"

        model = BoringModel(4)
        assert model.layer[0].weight.device.type == "meta"
        materialize_module(model)
        assert model.layer[0].weight.device.type == "cpu"

    mlp = MLP(4)
    materialize_module(mlp)
    assert mlp.layer[0].weight.device.type == "cpu"

    m = nn.Linear(in_features=1, out_features=1)
    assert m.weight.device.type == "cpu"

    with init_meta_context():
        m = nn.Linear(in_features=1, out_features=1)
        assert m.weight.device.type == "meta"

    m = nn.Linear(in_features=1, out_features=1)
    assert m.weight.device.type == "cpu"

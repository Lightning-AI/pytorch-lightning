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

from pytorch_lightning import Trainer
from pytorch_lightning.utilities import _PYTORCH_PRUNE_AVAILABLE
from tests.base import BoringModel

if _PYTORCH_PRUNE_AVAILABLE:
    from pytorch_lightning.callbacks import PruningCallback


@pytest.mark.skipif(not _PYTORCH_PRUNE_AVAILABLE, reason="PyTorch prung is needed for this test. ")
@pytest.mark.parametrize("parameters_to_prune", [False, True])
@pytest.mark.parametrize("use_global_unstructured", [False, True])
def test_pruning_callback(tmpdir, use_global_unstructured, parameters_to_prune):

    class PruningModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleDict()

            self.layer["mlp_1"] = nn.Linear(32, 32)
            self.layer["mlp_2"] = nn.Linear(32, 32)
            self.layer["mlp_3"] = nn.Linear(32, 32)
            self.layer["mlp_4"] = nn.Linear(32, 32)
            self.layer["mlp_5"] = nn.Linear(32, 2)

        def forward(self, x):
            m = self.layer
            x = m["mlp_1"](x)
            x = m["mlp_2"](x)
            x = m["mlp_3"](x)
            x = m["mlp_4"](x)
            return m["mlp_5"](x)

        def training_step(self, batch, batch_idx):
            output = self.forward(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

    model = PruningModel()
    model.validation_step = None
    model.test_step = None

    if parameters_to_prune:
        parameters_to_prune = [
            (model.layer["mlp_1"], "weight"),
            (model.layer["mlp_2"], "weight")
        ]

    else:
        parameters_to_prune = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_val_batches=2,
        max_epochs=10,
        callbacks=[
            PruningCallback(
                'l1_unstructured',
                parameters_to_prune=parameters_to_prune,
                amount=0.01,
                use_global_unstructured=use_global_unstructured,
            )
        ]
    )
    trainer.fit(model)
    _ = trainer.test(model)

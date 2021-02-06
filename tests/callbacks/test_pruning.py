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
import platform

import numpy as np
import pytest
import torch
from torch import nn

from pytorch_lightning import Trainer
from pytorch_lightning.utilities import _PYTORCH_PRUNE_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel

if _PYTORCH_PRUNE_AVAILABLE:
    import torch.nn.utils.prune as pytorch_prune

    from pytorch_lightning.callbacks import ModelPruning


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


def train_with_pruning_callback(
    tmpdir,
    parameters_to_prune,
    use_global_unstructured,
    accelerator=None,
    gpus=None,
    num_processes=None,
    use_custom_pruning_fn=False,
):
    # Skipped as currently not supported.
    # Todo: add support for custom pruning_fn function.
    if not use_global_unstructured and use_custom_pruning_fn:
        return

    model = PruningModel()
    model.validation_step = None
    model.test_step = None

    if parameters_to_prune:
        parameters_to_prune = [
            (model.layer["mlp_1"], "weight"),
            (model.layer["mlp_2"], "weight"),
        ]

    else:
        parameters_to_prune = None

    assert torch.sum(model.layer["mlp_2"].weight == 0) == 0

    class TestPruningMethod(pytorch_prune.BasePruningMethod):
        """Prune every other entry in a tensor
        """
        PRUNING_TYPE = 'unstructured'

        def compute_mask(self, t, default_mask):
            mask = default_mask.clone()
            mask.view(-1)[::2] = 0
            return mask

        @classmethod
        def apply(cls, module, name, amount):
            r"""Adds the forward pre-hook that enables pruning on the fly and
            the reparametrization of a tensor in terms of the original tensor
            and the pruning mask.

            Args:
                module (nn.Module): module containing the tensor to prune
                name (str): parameter name within ``module`` on which pruning
                    will act.
                amount (int or float): quantity of parameters to prune.
                    If ``float``, should be between 0.0 and 1.0 and represent the
                    fraction of parameters to prune. If ``int``, it represents the
                    absolute number of parameters to prune.
            """
            return super(TestPruningMethod, cls).apply(module, name, amount=amount)

    custom_pruning_fn = TestPruningMethod

    pruning_funcs_structured = [
        "ln_structured",
        "random_structured",
    ]

    pruning_funcs_unstructured = [
        "l1_unstructured",
        "random_unstructured",
    ]

    if use_global_unstructured:
        pruning_list = pruning_funcs_unstructured
    else:
        pruning_list = pruning_funcs_unstructured + pruning_funcs_structured

    rand_idx = np.random.randint(len(pruning_list))
    pruning_fn = pruning_list[rand_idx]

    model_pruning_args = {
        "pruning_fn": custom_pruning_fn if use_custom_pruning_fn else pruning_fn,
        "parameters_to_prune": parameters_to_prune,
        "amount": 0.3,
        "use_global_unstructured": use_global_unstructured,
    }

    if "unstructured" not in pruning_fn:
        model_pruning_args["pruning_dim"] = 0

    if pruning_fn == "ln_structured":
        model_pruning_args["pruning_norm"] = 1

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_val_batches=2,
        max_epochs=10,
        accelerator=accelerator,
        gpus=gpus,
        num_processes=num_processes,
        callbacks=ModelPruning(**model_pruning_args)
    )
    trainer.fit(model)
    _ = trainer.test(model)

    if accelerator is None:
        assert torch.sum(model.layer["mlp_2"].weight == 0) > 0


def test_with_pruning_callback_misconfiguration(tmpdir):
    model_pruning_args = {
        "parameter_names": ["chocolat"],
    }

    with pytest.raises(MisconfigurationException, match='provided parameter_names'):
        _ = ModelPruning(**model_pruning_args)

    model_pruning_args = {
        "parameter_names": ["weight"],
        "pruning_fn": model_pruning_args,
    }

    with pytest.raises(MisconfigurationException, match='pruning_fn is expected to be the str in'):
        _ = ModelPruning(**model_pruning_args)

    model_pruning_args = {
        "parameter_names": ["weight"],
        "pruning_fn": "random_structured",
    }

    with pytest.raises(MisconfigurationException, match='should be provided'):
        _ = ModelPruning(**model_pruning_args)

    model_pruning_args = {
        "parameter_names": ["weight"],
        "pruning_fn": "ln_structured",
        "pruning_dim": 0,
    }

    with pytest.raises(MisconfigurationException, match='requesting `ln_structured` pruning, the `pruning_norm`'):
        _ = ModelPruning(**model_pruning_args)


@pytest.mark.skipif(not _PYTORCH_PRUNE_AVAILABLE, reason="PyTorch prung is needed for this test. ")
@pytest.mark.parametrize("parameters_to_prune", [False, True])
@pytest.mark.parametrize("use_global_unstructured", [False, True])
@pytest.mark.parametrize("use_custom_pruning_fn", [False, True])
def test_pruning_callback(tmpdir, use_global_unstructured, parameters_to_prune, use_custom_pruning_fn):
    train_with_pruning_callback(
        tmpdir,
        parameters_to_prune,
        use_global_unstructured,
        accelerator=None,
        gpus=None,
        num_processes=1,
        use_custom_pruning_fn=use_custom_pruning_fn
    )


@pytest.mark.skipif(not _PYTORCH_PRUNE_AVAILABLE, reason="PyTorch prung is needed for this test. ")
@pytest.mark.parametrize("parameters_to_prune", [False, True])
@pytest.mark.parametrize("use_global_unstructured", [False, True])
@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest"
)
def test_pruning_callback_ddp(tmpdir, use_global_unstructured, parameters_to_prune):
    train_with_pruning_callback(
        tmpdir, parameters_to_prune, use_global_unstructured, accelerator="ddp", gpus=2, num_processes=0
    )


@pytest.mark.skipif(not _PYTORCH_PRUNE_AVAILABLE, reason="PyTorch prung is needed for this test. ")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_pruning_callback_ddp_spawn(tmpdir):
    train_with_pruning_callback(tmpdir, False, True, accelerator="ddp_spawn", gpus=2, num_processes=None)


@pytest.mark.skipif(not _PYTORCH_PRUNE_AVAILABLE, reason="PyTorch prung is needed for this test. ")
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_pruning_callback_ddp_cpu(tmpdir):
    train_with_pruning_callback(tmpdir, True, False, accelerator="ddp_cpu", gpus=None, num_processes=2)

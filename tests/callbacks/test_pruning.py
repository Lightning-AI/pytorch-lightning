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
from collections import OrderedDict

import pytest
import torch
import torch.nn.utils.prune as pytorch_prune
from torch import nn
from torch.nn import Sequential

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel


class TestModel(BoringModel):
    validation_step = None
    test_step = None

    def __init__(self):
        super().__init__()
        self.layer = Sequential(OrderedDict([
            ("mlp_1", nn.Linear(32, 32)),
            ("mlp_2", nn.Linear(32, 32)),
            ("mlp_3", nn.Linear(32, 2)),
        ]))


class TestPruningMethod(pytorch_prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def compute_mask(self, _, default_mask):
        mask = default_mask.clone()
        # Prune every other entry in a tensor
        mask.view(-1)[::2] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        return super(TestPruningMethod, cls).apply(module, name, amount=amount)


def train_with_pruning_callback(
    tmpdir,
    parameters_to_prune=False,
    use_global_unstructured=False,
    pruning_fn="l1_unstructured",
    use_lottery_ticket_hypothesis=False,
    accelerator=None,
    gpus=None,
    num_processes=1,
):
    model = TestModel()

    # Weights are random. None is 0
    assert torch.all(model.layer.mlp_2.weight != 0)

    pruning_kwargs = {"pruning_fn": pruning_fn, "amount": 0.3, "use_global_unstructured": use_global_unstructured, "use_lottery_ticket_hypothesis": use_lottery_ticket_hypothesis}
    if parameters_to_prune:
        pruning_kwargs["parameters_to_prune"] = [(model.layer.mlp_1, "weight"), (model.layer.mlp_2, "weight")]
    else:
        pruning_kwargs["parameter_names"] = ["weight"]
    if isinstance(pruning_fn, str) and pruning_fn.endswith("_structured"):
        pruning_kwargs["pruning_dim"] = 0
    if pruning_fn == "ln_structured":
        pruning_kwargs["pruning_norm"] = 1

    # Misconfiguration checks
    if isinstance(pruning_fn, str) and pruning_fn.endswith("_structured") and use_global_unstructured:
        with pytest.raises(MisconfigurationException, match="is supported with `use_global_unstructured=True`"):
            ModelPruning(**pruning_kwargs)
        return
    if ModelPruning.is_pruning_method(pruning_fn) and not use_global_unstructured:
        with pytest.raises(MisconfigurationException, match="currently only supported with"):
            ModelPruning(**pruning_kwargs)
        return

    pruning = ModelPruning(**pruning_kwargs)

    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        checkpoint_callback=False,
        logger=False,
        limit_train_batches=10,
        limit_val_batches=2,
        max_epochs=10,
        accelerator=accelerator,
        gpus=gpus,
        num_processes=num_processes,
        callbacks=pruning,
    )
    trainer.fit(model)
    trainer.test(model)

    if not accelerator:
        # Check some have been pruned
        assert torch.any(model.layer.mlp_2.weight == 0)


def test_pruning_misconfiguration():
    with pytest.raises(MisconfigurationException, match=r"chocolate isn't in \('weight', 'bias'\)"):
        ModelPruning(parameter_names=["chocolate"])
    with pytest.raises(MisconfigurationException, match=r"expected to be a str in \["):
        ModelPruning(pruning_fn={})  # noqa
    with pytest.raises(MisconfigurationException, match="should be provided"):
        ModelPruning(pruning_fn="random_structured")
    with pytest.raises(MisconfigurationException, match="requesting `ln_structured` pruning, the `pruning_norm`"):
        ModelPruning(pruning_fn="ln_structured", pruning_dim=0)


@pytest.mark.parametrize("parameters_to_prune", [False, True])
@pytest.mark.parametrize("use_global_unstructured", [False, True])
@pytest.mark.parametrize(
    "pruning_fn", ["l1_unstructured", "random_unstructured", "ln_structured", "random_structured", TestPruningMethod]
)
@pytest.mark.parametrize("use_lottery_ticket_hypothesis", [False, True])
def test_pruning_callback(tmpdir, use_global_unstructured, parameters_to_prune, pruning_fn, use_lottery_ticket_hypothesis):
    train_with_pruning_callback(
        tmpdir,
        parameters_to_prune=parameters_to_prune,
        use_global_unstructured=use_global_unstructured,
        pruning_fn=pruning_fn,
        use_lottery_ticket_hypothesis=use_lottery_ticket_hypothesis,
    )


@pytest.mark.parametrize("parameters_to_prune", [False, True])
@pytest.mark.parametrize("use_global_unstructured", [False, True])
@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", "0") == "1", reason="test should be run outside of pytest"
)
def test_pruning_callback_ddp(tmpdir, use_global_unstructured, parameters_to_prune):
    train_with_pruning_callback(
        tmpdir,
        parameters_to_prune=parameters_to_prune,
        use_global_unstructured=use_global_unstructured,
        accelerator="ddp",
        gpus=2,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_pruning_callback_ddp_spawn(tmpdir):
    train_with_pruning_callback(tmpdir, use_global_unstructured=True, accelerator="ddp_spawn", gpus=2)


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_pruning_callback_ddp_cpu(tmpdir):
    train_with_pruning_callback(tmpdir, parameters_to_prune=True, accelerator="ddp_cpu", num_processes=2)


# TODO: lottery ticket tests
# TODO: iterative pruning tests
# TODO: saving tests
# TODO: sparsity history and tracking
# TODO: allow resampling
# TODO: second chance

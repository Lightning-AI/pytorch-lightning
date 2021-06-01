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
import re
from collections import OrderedDict
from logging import INFO
from typing import Union

import pytest
import torch
import torch.nn.utils.prune as pytorch_prune
from torch import nn
from torch.nn import Sequential

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class TestModel(BoringModel):
    test_step = None

    def __init__(self):
        super().__init__()
        self.layer = Sequential(
            OrderedDict([
                ("mlp_1", nn.Linear(32, 32)),
                ("mlp_2", nn.Linear(32, 32, bias=False)),
                ("mlp_3", nn.Linear(32, 2)),
            ])
        )

    def training_step(self, batch, batch_idx):
        self.log("test", -batch_idx)
        return super().training_step(batch, batch_idx)


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

    pruning_kwargs = {
        "pruning_fn": pruning_fn,
        "amount": 0.3,
        "use_global_unstructured": use_global_unstructured,
        "use_lottery_ticket_hypothesis": use_lottery_ticket_hypothesis,
        "verbose": 1,
    }
    if parameters_to_prune:
        pruning_kwargs["parameters_to_prune"] = [(model.layer.mlp_1, "weight"), (model.layer.mlp_2, "weight")]
    else:
        if isinstance(pruning_fn, str) and pruning_fn.endswith("_structured"):
            pruning_kwargs["parameter_names"] = ["weight"]
        else:
            pruning_kwargs["parameter_names"] = ["weight", "bias"]
    if isinstance(pruning_fn, str) and pruning_fn.endswith("_structured"):
        pruning_kwargs["pruning_dim"] = 0
    if pruning_fn == "ln_structured":
        pruning_kwargs["pruning_norm"] = 1

    # Misconfiguration checks
    if isinstance(pruning_fn, str) and pruning_fn.endswith("_structured") and use_global_unstructured:
        with pytest.raises(MisconfigurationException, match="is supported with `use_global_unstructured=True`"):
            ModelPruning(**pruning_kwargs)
        return
    if ModelPruning._is_pruning_method(pruning_fn) and not use_global_unstructured:
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
        ModelPruning(pruning_fn="l1_unstructured", parameter_names=["chocolate"])
    with pytest.raises(MisconfigurationException, match=r"expected to be a str in \["):
        ModelPruning(pruning_fn={})  # noqa
    with pytest.raises(MisconfigurationException, match="should be provided"):
        ModelPruning(pruning_fn="random_structured")
    with pytest.raises(MisconfigurationException, match=r"must be any of \(0, 1, 2\)"):
        ModelPruning(pruning_fn="l1_unstructured", verbose=3)
    with pytest.raises(MisconfigurationException, match="requesting `ln_structured` pruning, the `pruning_norm`"):
        ModelPruning(pruning_fn="ln_structured", pruning_dim=0)


@pytest.mark.parametrize("parameters_to_prune", [False, True])
@pytest.mark.parametrize("use_global_unstructured", [False, True])
@pytest.mark.parametrize(
    "pruning_fn", ["l1_unstructured", "random_unstructured", "ln_structured", "random_structured", TestPruningMethod]
)
@pytest.mark.parametrize("use_lottery_ticket_hypothesis", [False, True])
def test_pruning_callback(
    tmpdir, use_global_unstructured: bool, parameters_to_prune: bool,
    pruning_fn: Union[str, pytorch_prune.BasePruningMethod], use_lottery_ticket_hypothesis: bool
):
    train_with_pruning_callback(
        tmpdir,
        parameters_to_prune=parameters_to_prune,
        use_global_unstructured=use_global_unstructured,
        pruning_fn=pruning_fn,
        use_lottery_ticket_hypothesis=use_lottery_ticket_hypothesis,
    )


@RunIf(special=True)
@pytest.mark.parametrize("parameters_to_prune", [False, True])
@pytest.mark.parametrize("use_global_unstructured", [False, True])
def test_pruning_callback_ddp(tmpdir, use_global_unstructured: bool, parameters_to_prune: bool):
    train_with_pruning_callback(
        tmpdir,
        parameters_to_prune=parameters_to_prune,
        use_global_unstructured=use_global_unstructured,
        accelerator="ddp",
        gpus=2,
    )


@RunIf(min_gpus=2, skip_windows=True)
def test_pruning_callback_ddp_spawn(tmpdir):
    train_with_pruning_callback(tmpdir, use_global_unstructured=True, accelerator="ddp_spawn", gpus=2)


@RunIf(skip_windows=True)
def test_pruning_callback_ddp_cpu(tmpdir):
    train_with_pruning_callback(tmpdir, parameters_to_prune=True, accelerator="ddp_cpu", num_processes=2)


@pytest.mark.parametrize("resample_parameters", (False, True))
def test_pruning_lth_callable(tmpdir, resample_parameters: bool):
    model = TestModel()

    class ModelPruningTestCallback(ModelPruning):
        lth_calls = 0

        def apply_lottery_ticket_hypothesis(self):
            super().apply_lottery_ticket_hypothesis()
            self.lth_calls += 1

            for d in self._original_layers.values():
                copy, names = d["data"], d["names"]
                for i, name in names:
                    curr, curr_name = self._parameters_to_prune[i]
                    assert name == curr_name
                    actual, expected = getattr(curr, name).data, getattr(copy, name).data
                    allclose = torch.allclose(actual, expected)
                    assert not allclose if self._resample_parameters else allclose

    pruning = ModelPruningTestCallback(
        "l1_unstructured", use_lottery_ticket_hypothesis=lambda e: bool(e % 2), resample_parameters=resample_parameters
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        checkpoint_callback=False,
        logger=False,
        limit_train_batches=10,
        limit_val_batches=2,
        max_epochs=5,
        callbacks=pruning,
    )
    trainer.fit(model)

    assert pruning.lth_calls == trainer.max_epochs // 2


@pytest.mark.parametrize("make_pruning_permanent", (False, True))
def test_multiple_pruning_callbacks(tmpdir, caplog, make_pruning_permanent: bool):
    model = TestModel()
    pruning_kwargs = {
        'parameters_to_prune': [(model.layer.mlp_1, "weight"), (model.layer.mlp_3, "weight")],
        'verbose': 2,
        "make_pruning_permanent": make_pruning_permanent
    }
    p1 = ModelPruning("l1_unstructured", amount=0.5, apply_pruning=lambda e: not e % 2, **pruning_kwargs)
    p2 = ModelPruning("random_unstructured", amount=0.25, apply_pruning=lambda e: e % 2, **pruning_kwargs)

    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        checkpoint_callback=False,
        logger=False,
        limit_train_batches=10,
        limit_val_batches=2,
        max_epochs=3,
        callbacks=[p1, p2],
    )
    with caplog.at_level(INFO):
        trainer.fit(model)

    actual = [m.strip() for m in caplog.messages]
    actual = [m for m in actual if m.startswith("Applied")]
    percentage = r"\(\d+(?:\.\d+)?%\)"
    expected = [
        rf"Applied `L1Unstructured`. Pruned: \d+\/1122 {percentage} -> \d+\/1122 {percentage}",
        rf"Applied `L1Unstructured` to `Linear\(in_features=32, out_features=32, bias=True\).weight` with amount=0.5. Pruned: 0 \(0.00%\) -> \d+ {percentage}",  # noqa: E501
        rf"Applied `L1Unstructured` to `Linear\(in_features=32, out_features=2, bias=True\).weight` with amount=0.5. Pruned: 0 \(0.00%\) -> \d+ {percentage}",  # noqa: E501
        rf"Applied `RandomUnstructured`. Pruned: \d+\/1122 {percentage} -> \d+\/1122 {percentage}",
        rf"Applied `RandomUnstructured` to `Linear\(in_features=32, out_features=32, bias=True\).weight` with amount=0.25. Pruned: \d+ {percentage} -> \d+ {percentage}",  # noqa: E501
        rf"Applied `RandomUnstructured` to `Linear\(in_features=32, out_features=2, bias=True\).weight` with amount=0.25. Pruned: \d+ {percentage} -> \d+ {percentage}",  # noqa: E501
        rf"Applied `L1Unstructured`. Pruned: \d+\/1122 {percentage} -> \d+\/1122 {percentage}",
        rf"Applied `L1Unstructured` to `Linear\(in_features=32, out_features=32, bias=True\).weight` with amount=0.5. Pruned: \d+ {percentage} -> \d+ {percentage}",  # noqa: E501
        rf"Applied `L1Unstructured` to `Linear\(in_features=32, out_features=2, bias=True\).weight` with amount=0.5. Pruned: \d+ {percentage} -> \d+ {percentage}",  # noqa: E501
    ]
    expected = [re.compile(s) for s in expected]
    assert all(regex.match(s) for s, regex in zip(actual, expected))

    filepath = str(tmpdir / "foo.ckpt")
    trainer.save_checkpoint(filepath)

    model.load_from_checkpoint(filepath, strict=False)
    has_pruning = hasattr(model.layer.mlp_1, "weight_orig")
    assert not has_pruning if make_pruning_permanent else has_pruning


@pytest.mark.parametrize("on_train_epoch_end", (False, True))
def test_permanent_when_model_is_saved_multiple_times(tmpdir, caplog, on_train_epoch_end):
    """
    When a model is saved multiple times and make_permanent=True, we need to
    make sure a copy is pruned and not the trained model if we want to continue
    with the same pruning buffers.
    """

    class TestPruning(ModelPruning):

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            super().on_save_checkpoint(trainer, pl_module, checkpoint)
            if not on_train_epoch_end:
                # these checks only work if pruning on `validation_epoch_end`
                # because `on_save_checkpoint` is called before `on_train_epoch_end`
                assert "layer.mlp_3.weight_orig" not in checkpoint["state_dict"]
                assert hasattr(pl_module.layer.mlp_3, "weight_orig")

    model = TestModel()
    pruning_callback = TestPruning(
        "random_unstructured",
        parameters_to_prune=[(model.layer.mlp_3, "weight")],
        verbose=1,
        make_pruning_permanent=True,
        prune_on_train_epoch_end=on_train_epoch_end,
    )
    ckpt_callback = ModelCheckpoint(monitor="test", save_top_k=2, save_last=True)
    trainer = Trainer(callbacks=[pruning_callback, ckpt_callback], max_epochs=3, progress_bar_refresh_rate=0)
    with caplog.at_level(INFO):
        trainer.fit(model)

    actual = [m.strip() for m in caplog.messages]
    actual = [m for m in actual if m.startswith("Applied")]
    percentage = r"\(\d+(?:\.\d+)?%\)"
    expected = [
        rf"Applied `RandomUnstructured`. Pruned: \d+\/66 {percentage} -> \d+\/66 {percentage}",
        rf"Applied `RandomUnstructured`. Pruned: \d+\/66 {percentage} -> \d+\/66 {percentage}",
        rf"Applied `RandomUnstructured`. Pruned: \d+\/66 {percentage} -> \d+\/66 {percentage}",
    ]
    expected = [re.compile(s) for s in expected]
    assert all(regex.match(s) for s, regex in zip(actual, expected))

    # removed on_train_end
    assert not hasattr(model.layer.mlp_3, "weight_orig")

    model.load_from_checkpoint(trainer.checkpoint_callback.kth_best_model_path)
    assert not hasattr(model.layer.mlp_3, "weight_orig")
    model.load_from_checkpoint(trainer.checkpoint_callback.last_model_path)
    assert not hasattr(model.layer.mlp_3, "weight_orig")

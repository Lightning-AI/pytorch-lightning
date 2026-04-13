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
import re
from collections import OrderedDict
from logging import INFO
from typing import Union

import pytest
import torch
import torch.nn.utils.prune as pytorch_prune
from torch import nn
from torch.nn import Sequential

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ModelPruning
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf


class TestModel(BoringModel):
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
        return super().apply(module, name, amount=amount)


def train_with_pruning_callback(
    tmp_path,
    parameters_to_prune=False,
    use_global_unstructured=False,
    pruning_fn="l1_unstructured",
    use_lottery_ticket_hypothesis=False,
    strategy="auto",
    accelerator="cpu",
    devices=1,
):
    seed_everything(1)
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
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        limit_train_batches=10,
        limit_val_batches=2,
        max_epochs=10,
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        callbacks=pruning,
    )
    trainer.fit(model)
    trainer.test(model)

    if not strategy:
        # Check some have been pruned
        assert torch.any(model.layer.mlp_2.weight == 0)


def test_pruning_misconfiguration():
    with pytest.raises(MisconfigurationException, match=r"chocolate isn't in \('weight', 'bias'\)"):
        ModelPruning(pruning_fn="l1_unstructured", parameter_names=["chocolate"])
    with pytest.raises(MisconfigurationException, match=r"expected to be a str in \["):
        ModelPruning(pruning_fn={})
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
    tmp_path,
    use_global_unstructured: bool,
    parameters_to_prune: bool,
    pruning_fn: Union[str, pytorch_prune.BasePruningMethod],
    use_lottery_ticket_hypothesis: bool,
):
    train_with_pruning_callback(
        tmp_path,
        parameters_to_prune=parameters_to_prune,
        use_global_unstructured=use_global_unstructured,
        pruning_fn=pruning_fn,
        use_lottery_ticket_hypothesis=use_lottery_ticket_hypothesis,
    )


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize("parameters_to_prune", [False, True])
@pytest.mark.parametrize("use_global_unstructured", [False, True])
def test_pruning_callback_ddp(tmp_path, parameters_to_prune, use_global_unstructured):
    train_with_pruning_callback(
        tmp_path,
        parameters_to_prune=parameters_to_prune,
        use_global_unstructured=use_global_unstructured,
        strategy="ddp",
        accelerator="gpu",
        devices=2,
    )


@RunIf(min_cuda_gpus=2, skip_windows=True)
def test_pruning_callback_ddp_spawn(tmp_path):
    train_with_pruning_callback(
        tmp_path, use_global_unstructured=True, strategy="ddp_spawn", accelerator="gpu", devices=2
    )


@RunIf(skip_windows=True)
def test_pruning_callback_ddp_cpu(tmp_path):
    train_with_pruning_callback(tmp_path, parameters_to_prune=True, strategy="ddp_spawn", accelerator="cpu", devices=2)


@pytest.mark.parametrize("resample_parameters", [False, True])
def test_pruning_lth_callable(tmp_path, resample_parameters):
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
                    # Check weight_orig if parameter is pruned, otherwise check the parameter directly
                    if hasattr(curr, name + "_orig"):
                        actual = getattr(curr, name + "_orig").data
                    else:
                        actual = getattr(curr, name).data
                    expected = getattr(copy, name).data
                    allclose = torch.allclose(actual.cpu(), expected)
                    assert not allclose if self._resample_parameters else allclose

    pruning = ModelPruningTestCallback(
        "l1_unstructured", use_lottery_ticket_hypothesis=lambda e: bool(e % 2), resample_parameters=resample_parameters
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        limit_train_batches=10,
        limit_val_batches=2,
        max_epochs=5,
        callbacks=pruning,
    )
    trainer.fit(model)

    assert pruning.lth_calls == trainer.max_epochs // 2


@pytest.mark.parametrize("make_pruning_permanent", [False, True])
def test_multiple_pruning_callbacks(tmp_path, caplog, make_pruning_permanent: bool):
    model = TestModel()
    pruning_kwargs = {
        "parameters_to_prune": [(model.layer.mlp_1, "weight"), (model.layer.mlp_3, "weight")],
        "verbose": 2,
        "make_pruning_permanent": make_pruning_permanent,
    }
    p1 = ModelPruning("l1_unstructured", amount=0.5, apply_pruning=lambda e: not e % 2, **pruning_kwargs)
    p2 = ModelPruning("random_unstructured", amount=0.25, apply_pruning=lambda e: e % 2, **pruning_kwargs)

    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
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
        rf"Applied `L1Unstructured`. Pruned: \d+\/1088 {percentage} -> \d+\/1088 {percentage}",
        rf"Applied `L1Unstructured` to `Linear\(in_features=32, out_features=32, bias=True\).weight` with amount=0.5. Pruned: 0 \(0.00%\) -> \d+ {percentage}",  # noqa: E501
        rf"Applied `L1Unstructured` to `Linear\(in_features=32, out_features=2, bias=True\).weight` with amount=0.5. Pruned: 0 \(0.00%\) -> \d+ {percentage}",  # noqa: E501
        rf"Applied `RandomUnstructured`. Pruned: \d+\/1088 {percentage} -> \d+\/1088 {percentage}",
        rf"Applied `RandomUnstructured` to `Linear\(in_features=32, out_features=32, bias=True\).weight` with amount=0.25. Pruned: \d+ {percentage} -> \d+ {percentage}",  # noqa: E501
        rf"Applied `RandomUnstructured` to `Linear\(in_features=32, out_features=2, bias=True\).weight` with amount=0.25. Pruned: \d+ {percentage} -> \d+ {percentage}",  # noqa: E501
        rf"Applied `L1Unstructured`. Pruned: \d+\/1088 {percentage} -> \d+\/1088 {percentage}",
        rf"Applied `L1Unstructured` to `Linear\(in_features=32, out_features=32, bias=True\).weight` with amount=0.5. Pruned: \d+ {percentage} -> \d+ {percentage}",  # noqa: E501
        rf"Applied `L1Unstructured` to `Linear\(in_features=32, out_features=2, bias=True\).weight` with amount=0.5. Pruned: \d+ {percentage} -> \d+ {percentage}",  # noqa: E501
    ]
    expected = [re.compile(s) for s in expected]
    assert all(regex.match(s) for s, regex in zip(actual, expected))

    filepath = str(tmp_path / "foo.ckpt")
    trainer.save_checkpoint(filepath)

    model.load_state_dict(torch.load(filepath, weights_only=True), strict=False)
    has_pruning = hasattr(model.layer.mlp_1, "weight_orig")
    assert not has_pruning if make_pruning_permanent else has_pruning


@pytest.mark.parametrize("prune_on_train_epoch_end", [False, True])
@pytest.mark.parametrize("save_on_train_epoch_end", [False, True])
def test_permanent_when_model_is_saved_multiple_times(
    tmp_path, caplog, prune_on_train_epoch_end, save_on_train_epoch_end
):
    """When a model is saved multiple times and make_permanent=True, we need to make sure a copy is pruned and not the
    trained model if we want to continue with the same pruning buffers."""
    if prune_on_train_epoch_end and save_on_train_epoch_end:
        pytest.xfail(
            "Pruning sets the `grad_fn` of the parameters so we can't save"
            " right after as pruning has not been made permanent"
        )

    class TestPruning(ModelPruning):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            had_buffers = hasattr(pl_module.layer.mlp_3, "weight_orig")
            super().on_save_checkpoint(trainer, pl_module, checkpoint)
            assert "layer.mlp_3.weight_orig" not in checkpoint["state_dict"]
            if had_buffers:
                assert hasattr(pl_module.layer.mlp_3, "weight_orig")

    model = TestModel()
    pruning_callback = TestPruning(
        "random_unstructured",
        parameters_to_prune=[(model.layer.mlp_3, "weight")],
        verbose=1,
        make_pruning_permanent=True,
        prune_on_train_epoch_end=prune_on_train_epoch_end,
    )
    ckpt_callback = ModelCheckpoint(
        monitor="test", save_top_k=2, save_last=True, save_on_train_epoch_end=save_on_train_epoch_end
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=False,
        callbacks=[pruning_callback, ckpt_callback],
        max_epochs=3,
        enable_progress_bar=False,
    )
    with caplog.at_level(INFO):
        trainer.fit(model)

    actual = [m.strip() for m in caplog.messages]
    actual = [m for m in actual if m.startswith("Applied")]
    percentage = r"\(\d+(?:\.\d+)?%\)"
    expected = [
        rf"Applied `RandomUnstructured`. Pruned: \d+\/64 {percentage} -> \d+\/64 {percentage}",
        rf"Applied `RandomUnstructured`. Pruned: \d+\/64 {percentage} -> \d+\/64 {percentage}",
        rf"Applied `RandomUnstructured`. Pruned: \d+\/64 {percentage} -> \d+\/64 {percentage}",
    ]
    expected = [re.compile(s) for s in expected]
    assert all(regex.match(s) for s, regex in zip(actual, expected))

    # removed on_train_end
    assert not hasattr(model.layer.mlp_3, "weight_orig")

    model = TestModel.load_from_checkpoint(trainer.checkpoint_callback.kth_best_model_path)
    assert not hasattr(model.layer.mlp_3, "weight_orig")
    model = TestModel.load_from_checkpoint(trainer.checkpoint_callback.last_model_path)
    assert not hasattr(model.layer.mlp_3, "weight_orig")


def test_sanitize_parameters_explicit_check():
    """Test the sanitize_parameters_to_prune method with various attribute types."""

    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 5))
            self.bias = nn.Parameter(torch.randn(5))
            self.some_bool = True
            self.some_tensor = torch.randn(3, 3)  # Regular tensor, not parameter
            self.some_string = "test"
            self.some_none = None

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.test_module = TestModule()

    model = TestModel()

    parameters_to_prune = ModelPruning.sanitize_parameters_to_prune(
        model,
        parameters_to_prune=(),
        parameter_names=["weight", "bias", "some_bool", "some_tensor", "some_string", "some_none"],
    )

    param_names_found = set()
    for module, param_name in parameters_to_prune:
        param = getattr(module, param_name)
        assert isinstance(param, nn.Parameter), f"Expected Parameter, got {type(param)}"
        param_names_found.add(param_name)

    assert "weight" in param_names_found
    assert "bias" in param_names_found
    assert "some_bool" not in param_names_found
    assert "some_tensor" not in param_names_found
    assert "some_string" not in param_names_found
    assert "some_none" not in param_names_found


def test_original_issue_reproduction():
    """Issue: https://github.com/Lightning-AI/pytorch-lightning/issues/10835."""

    class ProblematicModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = Sequential(
                OrderedDict([
                    ("mlp_1", nn.Linear(32, 32)),
                    ("mlp_2", nn.Linear(32, 2)),
                ])
            )
            # Add boolean attributes that would cause the original error
            self.layer.mlp_1.training = True
            self.layer.mlp_2.requires_grad = True

    model = ProblematicModel()

    parameters_to_prune = ModelPruning.sanitize_parameters_to_prune(
        model, parameters_to_prune=(), parameter_names=["weight", "bias", "training", "requires_grad"]
    )

    for module, param_name in parameters_to_prune:
        param = getattr(module, param_name)
        assert isinstance(param, nn.Parameter), f"Non-parameter found: {type(param)}"


def test_lottery_ticket_hypothesis_correctly_reset(tmp_path):
    """Test that lottery ticket hypothesis correctly resets unpruned weights to original values."""
    seed_everything(42)

    class LTHTestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(32, 2, bias=False)
            with torch.no_grad():
                # Initialize with a simple pattern for verification
                self.layer.weight.copy_(torch.arange(1, 65, dtype=torch.float32).reshape(2, 32))

    model = LTHTestModel()
    original_weights = model.layer.weight.data.clone()

    # Create a pruning callback that applies both pruning and LTH at epoch 1
    pruning_callback = ModelPruning(
        "l1_unstructured",
        parameters_to_prune=[(model.layer, "weight")],
        use_lottery_ticket_hypothesis=lambda epoch: epoch == 1,
        amount=0.5,
        verbose=0,  # Reduce verbosity
        make_pruning_permanent=False,
        apply_pruning=lambda epoch: epoch == 1,
    )

    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        limit_train_batches=5,
        limit_val_batches=1,
        max_epochs=2,
        accelerator="cpu",
        callbacks=pruning_callback,
    )
    trainer.fit(model)

    # After training with LTH applied, check that weight_orig was reset correctly
    assert hasattr(model.layer, "weight_mask"), "Pruning should have created weight_mask"
    assert hasattr(model.layer, "weight_orig"), "Pruning should have created weight_orig"

    weight_orig = getattr(model.layer, "weight_orig")
    assert torch.allclose(weight_orig, original_weights, atol=1e-6), (
        f"Lottery ticket hypothesis failed. weight_orig should be reset to original values.\n"
        f"Expected weight_orig: {original_weights}\n"
        f"Actual weight_orig: {weight_orig}\n"
        f"Max difference: {torch.max(torch.abs(weight_orig - original_weights))}"
    )


@pytest.mark.parametrize("pruning_amount", [0.1, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("model_type", ["simple", "complex"])
def test_sparsity_calculation(tmp_path, caplog, pruning_amount: float, model_type: str):
    """Test that the sparsity calculation fix correctly reports percentages."""

    class SimpleModel(BoringModel):
        """Simple model with 66 parameters (64 weight + 2 bias)."""

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(32, 2)  # 32*2 + 2 = 66 params

    class ComplexModel(BoringModel):
        """Complex model with multiple layers."""

        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(32, 64)  # 32*64 + 64 = 2112 params
            self.layer2 = nn.Linear(64, 2)  # 64*2 + 2 = 130 params
            # Total: 2112 + 130 = 2242 params (but only layer1 will be pruned)
            # layer1 params: 2112

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            return self.layer2(x)

    if model_type == "simple":
        model = SimpleModel()
        expected_total_params = 66
        parameters_to_prune = None
    else:
        model = ComplexModel()
        expected_total_params = 2112
        parameters_to_prune = [(model.layer1, "weight"), (model.layer1, "bias")]

    pruning = ModelPruning(
        pruning_fn="l1_unstructured",
        parameters_to_prune=parameters_to_prune,
        amount=pruning_amount,
        verbose=1,
        use_global_unstructured=True,
    )

    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        limit_train_batches=1,
        max_epochs=1,
        accelerator="cpu",
        callbacks=[pruning],
    )

    with caplog.at_level(INFO):
        trainer.fit(model)

    sparsity_logs = [msg for msg in caplog.messages if "Applied `L1Unstructured`. Pruned:" in msg]
    assert len(sparsity_logs) == 1, f"Expected 1 sparsity log, got {len(sparsity_logs)}"
    sparsity_log = sparsity_logs[0]
    pattern = r"Applied `L1Unstructured`\. Pruned: \d+/(\d+) \(\d+\.\d+%\) -> (\d+)/(\d+) \((\d+\.\d+)%\)"
    match = re.search(pattern, sparsity_log)
    assert match, f"Could not parse sparsity log: {sparsity_log}"

    total_params_before = int(match.group(1))
    pruned_count = int(match.group(2))
    total_params_after = int(match.group(3))
    sparsity_percentage = float(match.group(4))
    assert total_params_before == expected_total_params, (
        f"Total parameter count mismatch for {model_type} model. "
        f"Expected {expected_total_params}, got {total_params_before}"
    )
    assert total_params_after == expected_total_params, (
        f"Total parameter count should be consistent. Before: {total_params_before}, After: {total_params_after}"
    )

    # Verify sparsity percentage is approximately correct
    expected_sparsity = pruning_amount * 100
    tolerance = 5.0
    assert abs(sparsity_percentage - expected_sparsity) <= tolerance

    # Verify the number of pruned parameters is reasonable
    expected_pruned_count = int(expected_total_params * pruning_amount)
    pruned_tolerance = max(1, int(expected_total_params * 0.05))
    assert abs(pruned_count - expected_pruned_count) <= pruned_tolerance

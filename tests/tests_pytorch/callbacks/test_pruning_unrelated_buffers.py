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
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ModelPruning
from lightning.pytorch.demos.boring_classes import BoringModel


def _callback(model, **kwargs):
    return ModelPruning(
        "l1_unstructured",
        parameters_to_prune=[(model.layer, "weight")],
        amount=0.5,
        use_lottery_ticket_hypothesis=False,
        **kwargs,
    )


def _prune_parameter(module, name):
    mask = torch.ones_like(getattr(module, name))
    mask.reshape(-1)[::2] = 0
    prune.custom_from_mask(module, name, mask)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("paired_buffers", [False, True])
@pytest.mark.parametrize("pruned_names", [(), ("weight",), ("weight", "bias")])
def test_checkpoint_preserves_unrelated_masks(nested, paired_buffers, pruned_names):
    model = BoringModel()
    owner = model.layer if nested else model
    owner.register_buffer("causal_mask", torch.ones(3, 3, dtype=torch.bool).triu(1))
    if paired_buffers:
        # A matching name alone is not evidence of a pruning reparameterization.
        owner.register_buffer("activation_mask", torch.tensor([0.0, 1.0]))
        owner.register_buffer("activation_orig", torch.tensor([2.0, 3.0]))
    expected = deepcopy(model)
    for name in pruned_names:
        _prune_parameter(model.layer, name)
        _prune_parameter(expected.layer, name)
        prune.remove(expected.layer, name)
    live_state = {name: value.clone() for name, value in model.state_dict().items()}
    live_hooks = dict(model.layer._forward_pre_hooks)

    checkpoint = _callback(model)._make_pruning_permanent_on_state_dict(model)

    torch.testing.assert_close(checkpoint, expected.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), live_state, rtol=0, atol=0)
    assert model.layer._forward_pre_hooks == live_hooks
    assert all(value.device.type == "cpu" for value in checkpoint.values())
    expected.load_state_dict(checkpoint, strict=True)


@pytest.mark.parametrize("has_unrelated_mask", [False, True])
@pytest.mark.parametrize("repeat_pruning", [False, True])
def test_checkpoint_materializes_each_shared_module_path(has_unrelated_mask, repeat_pruning):
    model = BoringModel()
    model.shared_layer = model.layer
    if has_unrelated_mask:
        model.register_buffer("attention_mask", torch.tensor([True, False]))
    expected = deepcopy(model)
    for _ in range(1 + int(repeat_pruning)):
        _prune_parameter(model.layer, "weight")
        _prune_parameter(expected.layer, "weight")
    prune.remove(expected.layer, "weight")

    checkpoint = _callback(model)._make_pruning_permanent_on_state_dict(model)

    torch.testing.assert_close(checkpoint, expected.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["layer.weight"], checkpoint["shared_layer.weight"])
    assert prune.is_pruned(model.layer)
    expected.load_state_dict(checkpoint, strict=True)


@pytest.mark.parametrize("has_unrelated_mask", [False, True])
def test_checkpoint_materializes_root_parameter(has_unrelated_mask):
    model = BoringModel()
    model.register_parameter("scale", nn.Parameter(torch.tensor([2.0, 3.0])))
    if has_unrelated_mask:
        model.register_buffer("attention_mask", torch.tensor([True, False]))
    expected = deepcopy(model)
    _prune_parameter(model, "scale")
    _prune_parameter(expected, "scale")
    prune.remove(expected, "scale")

    checkpoint = _callback(model)._make_pruning_permanent_on_state_dict(model)

    torch.testing.assert_close(checkpoint, expected.state_dict(), rtol=0, atol=0)
    assert "scale" in checkpoint
    assert "scale_orig" in model.state_dict()


class _CausalAttentionModel(BoringModel):
    def __init__(self, nested=False):
        super().__init__()
        self.save_hyperparameters()
        self.attention = nn.MultiheadAttention(8, 2, batch_first=True)
        self.layer = nn.Linear(8, 2)
        self.nested = nested
        owner = self.attention if nested else self
        owner.register_buffer("causal_mask", torch.ones(4, 4, dtype=torch.bool).triu(1))

    @property
    def attention_mask(self):
        return (self.attention if self.nested else self).causal_mask

    def forward(self, x):
        x = x.reshape(-1, 4, 8)
        x, _ = self.attention(x, x, x, attn_mask=self.attention_mask, need_weights=False)
        return self.layer(x.mean(dim=1))

    def train_dataloader(self):
        return DataLoader(torch.linspace(-1, 1, 128).reshape(4, 32), batch_size=2)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.03, momentum=0.8)


def _trainer(tmp_path, **kwargs):
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("epochs", [1, 2])
def test_trainer_saves_pruned_attention_model(tmp_path, nested, epochs):
    seed_everything(17)
    model = _CausalAttentionModel(nested)
    original_mask = model.attention_mask.clone()
    checkpoint = ModelCheckpoint(save_last=True)
    trainer = _trainer(tmp_path, max_epochs=epochs, callbacks=[_callback(model), checkpoint])

    trainer.fit(model)

    saved = torch.load(checkpoint.last_model_path, weights_only=False)
    restored = _CausalAttentionModel.load_from_checkpoint(checkpoint.last_model_path)
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored.attention_mask, original_mask, rtol=0, atol=0)
    assert "layer.weight_orig" not in saved["state_dict"]
    assert "layer.weight_mask" not in saved["state_dict"]
    assert (restored.layer.weight == 0).sum() >= restored.layer.weight.numel() // 2
    inputs = next(iter(model.train_dataloader()))
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    resumed = _trainer(tmp_path / "resumed", max_epochs=epochs + 1, enable_checkpointing=False)
    resumed.fit(restored, ckpt_path=checkpoint.last_model_path)
    assert resumed.global_step == 2 * (epochs + 1)
    torch.testing.assert_close(restored.attention_mask, original_mask, rtol=0, atol=0)
    assert torch.isfinite(restored(inputs)).all()


@pytest.mark.parametrize("nested", [False, True])
def test_export_does_not_change_next_pruned_training_update(nested):
    seed_everything(29)
    model = _CausalAttentionModel(nested)
    expected = deepcopy(model)
    for network in (model, expected):
        _prune_parameter(network.layer, "weight")
    optimizers = [network.configure_optimizers() for network in (model, expected)]
    for step, inputs in enumerate(model.train_dataloader()):
        for network, optimizer in zip((model, expected), optimizers):
            optimizer.zero_grad()
            network(inputs).square().mean().backward()
            optimizer.step()
        if step == 0:
            checkpoint = _callback(model)._make_pruning_permanent_on_state_dict(model)
            assert "layer.weight" in checkpoint
            assert prune.is_pruned(model.layer)
    torch.testing.assert_close(model.state_dict(), expected.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizers[0].state_dict(), optimizers[1].state_dict(), rtol=0, atol=0)
    for actual, reference in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0)

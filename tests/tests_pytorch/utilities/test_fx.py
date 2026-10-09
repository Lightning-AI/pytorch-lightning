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
import copy
import operator
from unittest import mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.fx import from_fx, to_unfx


class _Conv(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.weight.t() + self.bias)


class _HomogeneousModel(LightningModule):
    def __init__(self, in_channels: int = 3, hidden: int = 4, out_channels: int = 2) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.conv = _Conv(in_channels, hidden)
        self.lin = nn.Linear(hidden, out_channels)
        self.register_buffer("offset", torch.zeros(()), persistent=False)

    def forward(self, x, edge_index):
        return self.lin(self.conv(x, edge_index))

    def training_step(self, batch, batch_idx):
        x, edge_index = batch
        return self(x, edge_index).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _graph_module(model: _HomogeneousModel) -> torch.fx.GraphModule:
    # Mirrors the shape of PyG's to_hetero() output: per-edge-type submodules under dotted call_module targets
    root = nn.Module()
    root.conv = nn.ModuleDict({"a__b": copy.deepcopy(model.conv), "a__c": copy.deepcopy(model.conv)})
    root.lin = model.lin
    root.scale = nn.Parameter(torch.ones(()))
    root.register_buffer("offset", model.offset, persistent=False)
    root.dim = 1

    graph = torch.fx.Graph()
    x_dict = graph.placeholder("x_dict")
    edge_index_dict = graph.placeholder("edge_index_dict")
    x_a = graph.call_function(operator.getitem, (x_dict, "a"))
    e_b = graph.call_function(operator.getitem, (edge_index_dict, "a__b"))
    e_c = graph.call_function(operator.getitem, (edge_index_dict, "a__c"))
    h_b = graph.call_module("conv.a__b", (x_a, e_b))
    h_c = graph.call_module("conv.a__c", (x_a, e_c))
    h = graph.call_function(torch.add, (h_b, h_c))
    h = graph.call_function(torch.mul, (h, graph.get_attr("scale")))
    h = graph.call_function(torch.add, (h, graph.get_attr("offset")))
    out = graph.call_module("lin", (h,))
    graph.get_attr("dim")
    graph.output(out)
    return torch.fx.GraphModule(root, graph)


def _inputs():
    x = {"a": torch.randn(5, 3)}
    edge_index = {"a__b": torch.tensor([[0, 1], [1, 0]]), "a__c": torch.tensor([[0, 1], [1, 0]])}
    return x, edge_index


class _GraphDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return _inputs()


class _TensorDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.randn(5, 3), torch.tensor([[0, 1], [1, 0]])


def _trainer_kwargs(tmp_path):
    return {
        "default_root_dir": tmp_path,
        "fast_dev_run": True,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }


def test_from_fx_transfers_graph_and_keeps_the_module():
    model = _HomogeneousModel()
    original_conv = model.conv
    graph_module = _graph_module(model)
    x, edge_index = _inputs()
    expected = graph_module(x, edge_index)

    converted = from_fx(model, graph_module)

    assert converted is model
    assert isinstance(model, LightningModule)
    assert isinstance(model, _HomogeneousModel)
    assert model.conv is not original_conv
    assert model.conv.a__b is graph_module.get_submodule("conv.a__b")
    assert model.dim == 1
    assert torch.equal(model(x, edge_index), expected)
    assert set(model.state_dict()) == {
        "conv.a__b.weight",
        "conv.a__b.bias",
        "conv.a__c.weight",
        "conv.a__c.bias",
        "lin.weight",
        "lin.bias",
        "scale",
    }
    assert "offset" in model._non_persistent_buffers_set
    assert model.hparams["in_channels"] == 3


def test_from_fx_trainer_fit(tmp_path):
    model = _HomogeneousModel()
    original_conv = model.conv
    from_fx(model, _graph_module(model))

    trainer = Trainer(**_trainer_kwargs(tmp_path))
    trainer.fit(model, train_dataloaders=DataLoader(_GraphDataset(), batch_size=None))

    assert model.conv.a__b.weight.grad is not None
    assert original_conv.weight.grad is None


def test_from_fx_in_configure_model(tmp_path):
    class _Model(_HomogeneousModel):
        def configure_model(self):
            from_fx(self, _graph_module(self))

    model = _Model()
    trainer = Trainer(**_trainer_kwargs(tmp_path))
    trainer.fit(model, train_dataloaders=DataLoader(_GraphDataset(), batch_size=None))

    assert model.conv.a__b.weight.grad is not None


def test_to_unfx_restores_original_model():
    model = _HomogeneousModel()
    original_conv = model.conv
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    expected = model(x, edge_index)

    from_fx(model, _graph_module(model))
    restored = to_unfx(model)

    assert restored is model
    assert model.conv is original_conv
    assert model._fx_ctx is None
    assert torch.equal(model(x, edge_index), expected)


def test_to_unfx_requires_converted_module():
    model = _HomogeneousModel()

    with pytest.raises(ValueError, match="required to be a converted LightningModule"):
        to_unfx(model)


def test_from_fx_guards():
    model = _HomogeneousModel()
    graph_module = _graph_module(model)

    with pytest.raises(TypeError, match="`module` must be a `LightningModule`"):
        from_fx(nn.Linear(2, 2), graph_module)
    with pytest.raises(TypeError, match="`graph_module` must be a `torch.fx.GraphModule`"):
        from_fx(model, nn.Linear(2, 2))

    model._compiler_ctx = {"compiler": "dynamo"}
    with pytest.raises(RuntimeError, match="is compiled with `torch.compile"):
        from_fx(model, graph_module)
    model._compiler_ctx = None

    from_fx(model, graph_module)
    with pytest.raises(RuntimeError, match="already converted"):
        from_fx(model, graph_module)


def test_from_fx_restores_when_the_graph_is_incomplete():
    model = _HomogeneousModel()
    graph_module = _graph_module(model)
    del graph_module._modules["conv"]

    with pytest.raises(RuntimeError, match="submodule `conv.a__b`"):
        from_fx(model, graph_module)

    assert model._fx_ctx is None
    assert isinstance(model.conv, _Conv)
    assert set(model.state_dict()) == {"conv.weight", "conv.bias", "lin.weight", "lin.bias"}


def test_from_fx_after_training_raises(tmp_path):
    model = _HomogeneousModel()
    trainer = Trainer(**_trainer_kwargs(tmp_path))
    trainer.fit(model, train_dataloaders=DataLoader(_TensorDataset(), batch_size=None))

    with pytest.raises(RuntimeError, match="must be called before the Trainer sets up the model"):
        from_fx(model, _graph_module(model))


def test_from_fx_error_message_for_graph_module(tmp_path):
    model = _HomogeneousModel()
    graph_module = _graph_module(model)
    trainer = Trainer(**_trainer_kwargs(tmp_path))

    with pytest.raises(TypeError, match="lightning.pytorch.utilities.fx.from_fx"):
        trainer.fit(graph_module)


def test_from_fx_lazy_parameters():
    class _LazyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.LazyLinear(4)

        def forward(self, x, edge_index):
            return torch.relu(self.linear(x))

    class _LazyModel(_HomogeneousModel):
        def __init__(self):
            super().__init__()
            self.conv = _LazyConv()

    model = _LazyModel()
    from_fx(model, _graph_module(model))

    weight = model.conv.a__b.linear.weight
    assert torch.nn.parameter.is_lazy(weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x, edge_index = _inputs()
    model(x, edge_index).sum().backward()
    optimizer.step()

    assert not torch.nn.parameter.is_lazy(model.conv.a__b.linear.weight)
    assert model.conv.a__b.linear.weight is weight


@mock.patch("lightning.pytorch.trainer.call._call_and_handle_interrupt")
def test_from_fx_ddp(_, tmp_path, mps_count_0):
    model = _HomogeneousModel()
    from_fx(model, _graph_module(model))

    trainer = Trainer(strategy="ddp", **_trainer_kwargs(tmp_path))
    trainer.fit(model)

    assert trainer.strategy.lightning_module is model
    assert trainer.model is model


def test_from_fx_with_pyg_to_hetero(tmp_path):
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import SAGEConv, to_hetero

    class _Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), 4)
            self.conv2 = SAGEConv((-1, -1), 2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

        def training_step(self, batch, batch_idx):
            x_dict, edge_index_dict = batch
            return self(x_dict, edge_index_dict)["a"].pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    class _PyGDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            x_dict = {"a": torch.randn(5, 3)}
            edge_index_dict = {("a", "to", "a"): torch.tensor([[0, 1], [1, 0]])}
            return x_dict, edge_index_dict

    model = _Model()
    metadata = (["a"], [("a", "to", "a")])
    from_fx(model, to_hetero(model, metadata))

    assert any(key.startswith("conv1.a__to__a.") for key in model.state_dict())

    trainer = Trainer(**_trainer_kwargs(tmp_path))
    trainer.fit(model, train_dataloaders=DataLoader(_PyGDataset(), batch_size=None))

    assert all(p.grad is not None for name, p in model.named_parameters() if name.startswith("conv1."))

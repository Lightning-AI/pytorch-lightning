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
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.core.memory import ModelSummary, UNKNOWN_SIZE
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_9
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.advanced_models import ParityModuleRNN
from tests.helpers.runif import RunIf


class EmptyModule(LightningModule):
    """A module that has no layers"""

    def __init__(self):
        super().__init__()
        self.parameter = torch.rand(3, 3, requires_grad=True)
        self.example_input_array = torch.zeros(1, 2, 3, 4, 5)

    def forward(self, *args, **kwargs):
        return {"loss": self.parameter.sum()}


class PreCalculatedModel(BoringModel):
    """A model with precalculated total params size in MB for FP16 and FP32."""

    def __init__(self, precision: int = 32):
        super().__init__()
        # 32K params
        self.layer = nn.Linear(32, 1000, bias=False)
        # 218K params
        self.layer1 = nn.Linear(1000, 218, bias=False)
        # calculate model size based on precision.
        self.pre_calculated_model_size = 1.0 / (32 / precision)

    def forward(self, x):
        x = self.layer(x)
        return self.layer1(x)


class UnorderedModel(LightningModule):
    """A model in which the layers not defined in order of execution"""

    def __init__(self):
        super().__init__()
        # note: the definition order is intentionally scrambled for this test
        self.layer2 = nn.Linear(10, 2)
        self.combine = nn.Linear(7, 9)
        self.layer1 = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        # this layer is unused, therefore input-/output shapes are unknown
        self.unused = nn.Conv2d(1, 1, 1)

        self.example_input_array = (torch.rand(2, 3), torch.rand(2, 10))

    def forward(self, x, y):
        out1 = self.layer1(x)
        out2 = self.layer2(y)
        out = self.relu(torch.cat((out1, out2), 1))
        out = self.combine(out)
        return out


class MixedDtypeModel(LightningModule):
    """The parameters and inputs of this model have different dtypes."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 20)  # expects dtype long as input
        self.reduce = nn.Linear(20, 1)  # dtype: float
        self.example_input_array = torch.tensor([[0, 2, 1], [3, 5, 3]])  # dtype: long

    def forward(self, x):
        return self.reduce(self.embed(x))


class PartialScriptModel(LightningModule):
    """A model which contains scripted layers."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.jit.script(nn.Linear(5, 3))
        self.layer2 = nn.Linear(3, 2)
        self.example_input_array = torch.rand(2, 5)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class LazyModel(LightningModule):
    """A model which contains lazy layers with unintialized parameters."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.LazyLinear(5)
        self.layer2 = nn.LazyLinear(2)

    def forward(self, inp):
        return self.layer2(self.layer1(inp))


class DeepNestedModel(LightningModule):
    """A model with deep nested layers."""

    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(5, 5),
            nn.Sequential(
                nn.Linear(5, 5),
                nn.Sequential(
                    nn.Linear(5, 5),
                    nn.Sequential(nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 3)))),
                ),
            ),
        )
        self.branch2 = nn.Linear(5, 10)
        self.head = UnorderedModel()
        self.example_input_array = torch.rand(2, 5)

    def forward(self, inp):
        return self.head(self.branch1(inp), self.branch2(inp))


def test_invalid_weights_summmary():
    """Test that invalid value for weights_summary raises an error."""
    with pytest.raises(MisconfigurationException, match="`mode` can be None, .* got temp"):
        UnorderedModel().summarize(mode="temp")

    with pytest.raises(MisconfigurationException, match="`weights_summary` can be None, .* got temp"):
        Trainer(weights_summary="temp")


@pytest.mark.parametrize("mode", ["full", "top"])
def test_empty_model_summary_shapes(mode: str):
    """Test that the summary works for models that have no submodules."""
    model = EmptyModule()
    summary = model.summarize(mode=mode)
    assert summary.in_sizes == []
    assert summary.out_sizes == []
    assert summary.param_nums == []


@RunIf(min_gpus=1)
@pytest.mark.parametrize("mode", ["full", "top"])
@pytest.mark.parametrize(
    ["device"],
    [pytest.param(torch.device("cpu")), pytest.param(torch.device("cuda", 0)), pytest.param(torch.device("cuda", 0))],
)
def test_linear_model_summary_shapes(device, mode):
    """Test that the model summary correctly computes the input- and output shapes."""
    model = UnorderedModel().to(device)
    model.train()
    summary = model.summarize(mode=mode)
    assert summary.in_sizes == [[2, 10], [2, 7], [2, 3], [2, 7], UNKNOWN_SIZE]  # layer 2  # combine  # layer 1  # relu
    assert summary.out_sizes == [[2, 2], [2, 9], [2, 5], [2, 7], UNKNOWN_SIZE]  # layer 2  # combine  # layer 1  # relu
    assert model.training
    assert model.device == device


def test_mixed_dtype_model_summary():
    """Test that the model summary works with models that have mixed input- and parameter dtypes."""
    model = MixedDtypeModel()
    summary = model.summarize()
    assert summary.in_sizes == [[2, 3], [2, 3, 20]]  # embed  # reduce
    assert summary.out_sizes == [[2, 3, 20], [2, 3, 1]]  # embed  # reduce


@pytest.mark.parametrize("max_depth", [-1, 0])
def test_hooks_removed_after_summarize(max_depth):
    """Test that all hooks were properly removed after summary, even ones that were not run."""
    model = UnorderedModel()
    summary = ModelSummary(model, max_depth=max_depth)
    # hooks should be removed
    for _, layer in summary.summarize().items():
        handle = layer._hook_handle
        assert handle.id not in handle.hooks_dict_ref()


@pytest.mark.parametrize("mode", ["full", "top"])
def test_rnn_summary_shapes(mode):
    """Test that the model summary works for RNNs."""
    model = ParityModuleRNN()

    b = 3
    t = 5
    i = model.rnn.input_size
    h = model.rnn.hidden_size
    o = model.linear_out.out_features

    model.example_input_array = torch.zeros(b, t, 10)

    summary = model.summarize(mode=mode)
    assert summary.in_sizes == [[b, t, i], [b, t, h]]  # rnn  # linear
    assert summary.out_sizes == [[[b, t, h], [[1, b, h], [1, b, h]]], [b, t, o]]  # rnn  # linear


@pytest.mark.parametrize("mode", ["full", "top"])
def test_summary_parameter_count(mode):
    """Test that the summary counts the number of parameters in every submodule."""
    model = UnorderedModel()
    summary = model.summarize(mode=mode)
    assert summary.param_nums == [
        model.layer2.weight.numel() + model.layer2.bias.numel(),
        model.combine.weight.numel() + model.combine.bias.numel(),
        model.layer1.weight.numel() + model.layer1.bias.numel(),
        0,  # ReLU
        model.unused.weight.numel() + model.unused.bias.numel(),
    ]


@pytest.mark.parametrize("mode", ["full", "top"])
def test_summary_layer_types(mode):
    """Test that the summary displays the layer names correctly."""
    model = UnorderedModel()
    summary = model.summarize(mode=mode)
    assert summary.layer_types == ["Linear", "Linear", "Linear", "ReLU", "Conv2d"]


@pytest.mark.parametrize("mode", ["full", "top"])
def test_summary_with_scripted_modules(mode):
    model = PartialScriptModel()
    summary = model.summarize(mode=mode)
    assert summary.layer_types == ["RecursiveScriptModule", "Linear"]
    assert summary.in_sizes == [UNKNOWN_SIZE, [2, 3]]
    assert summary.out_sizes == [UNKNOWN_SIZE, [2, 2]]


@pytest.mark.parametrize("mode", ["full", "top"])
@pytest.mark.parametrize(
    ["example_input", "expected_size"],
    [
        pytest.param([], UNKNOWN_SIZE),
        pytest.param((1, 2, 3), [UNKNOWN_SIZE] * 3),
        pytest.param(torch.tensor(0), UNKNOWN_SIZE),
        pytest.param(dict(tensor=torch.zeros(1, 2, 3)), UNKNOWN_SIZE),
        pytest.param(torch.zeros(2, 3, 4), [2, 3, 4]),
        pytest.param([torch.zeros(2, 3), torch.zeros(4, 5)], [[2, 3], [4, 5]]),
        pytest.param((torch.zeros(2, 3), torch.zeros(4, 5)), [[2, 3], [4, 5]]),
    ],
)
def test_example_input_array_types(example_input, expected_size, mode):
    """Test the types of example inputs supported for display in the summary."""

    class DummyModule(nn.Module):
        def forward(self, *args, **kwargs):
            return None

    class DummyLightningModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = DummyModule()

        # this LightningModule and submodule accept any type of input
        def forward(self, *args, **kwargs):
            return self.layer(*args, **kwargs)

    model = DummyLightningModule()
    model.example_input_array = example_input
    summary = model.summarize(mode=mode)
    assert summary.in_sizes == [expected_size]


@pytest.mark.parametrize("mode", ["full", "top"])
def test_model_size(mode):
    """Test model size is calculated correctly."""
    model = PreCalculatedModel()
    summary = model.summarize(mode=mode)
    assert model.pre_calculated_model_size == summary.model_size


@pytest.mark.parametrize("mode", ["full", "top"])
def test_empty_model_size(mode):
    """Test empty model size is zero."""
    model = EmptyModule()
    summary = model.summarize(mode=mode)
    assert 0.0 == summary.model_size


@RunIf(min_gpus=1, amp_native=True)
def test_model_size_precision(tmpdir):
    """Test model size for half and full precision."""
    model = PreCalculatedModel()

    # fit model
    trainer = Trainer(default_root_dir=tmpdir, gpus=1, max_steps=1, max_epochs=1, precision=32)
    trainer.fit(model)
    summary = model.summarize()
    assert model.pre_calculated_model_size == summary.model_size


@RunIf(min_torch="1.8")
def test_lazy_model_summary():
    """Test that the model summary can work with lazy layers."""
    lazy_model = LazyModel()
    summary = ModelSummary(lazy_model)

    with pytest.warns(
        UserWarning,
        match=r"A layer with UninitializedParameter was found. "
        r"Thus, the total number of parameters detected may be inaccurate.",
    ):
        if _TORCH_GREATER_EQUAL_1_9:
            assert summary.total_parameters == 0
            assert summary.trainable_parameters == 0
        else:
            # bug in 1.8: the bias of a LazyLinear layer is initialized!
            # https://github.com/pytorch/pytorch/issues/58350
            assert summary.total_parameters == 7
            assert summary.trainable_parameters == 7


def test_max_depth_equals_mode_interface():
    """Test model.summarize(full/top) interface mapping matches max_depth"""
    model = DeepNestedModel()

    summary_top = model.summarize(mode="top")
    summary_0 = model.summarize(max_depth=1)
    assert str(summary_top) == str(summary_0)

    summary_full = model.summarize(mode="full")
    summary_minus1 = model.summarize(max_depth=-1)
    assert str(summary_full) == str(summary_minus1)


@pytest.mark.parametrize("max_depth", [-1, 0, 1, 3, 999])
def test_max_depth_param(max_depth):
    """Test that only the modules up to the desired depth are shown"""
    model = DeepNestedModel()
    summary = ModelSummary(model, max_depth=max_depth)
    for lname in summary.layer_names:
        if max_depth >= 0:
            assert lname.count(".") < max_depth


@pytest.mark.parametrize("max_depth", [-99, -2, "invalid"])
def test_raise_invalid_max_depth_value(max_depth):
    with pytest.raises(ValueError, match=f"`max_depth` can be -1, 0 or > 0, got {max_depth}"):
        DeepNestedModel().summarize(max_depth=max_depth)

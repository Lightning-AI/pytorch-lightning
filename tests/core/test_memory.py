import pytest
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from pytorch_lightning.core.memory import UNKNOWN_SIZE, ModelSummary
from tests.base.models import ParityModuleRNN


class EmptyModule(LightningModule):
    """ A module that has no layers """

    def __init__(self):
        super().__init__()
        self.parameter = torch.rand(3, 3, requires_grad=True)
        self.example_input_array = torch.zeros(1, 2, 3, 4, 5)

    def forward(self, *args, **kwargs):
        return {'loss': self.parameter.sum()}


class UnorderedModel(LightningModule):
    """ A model in which the layers not defined in order of execution """

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
    """ The parameters and inputs of this model have different dtypes. """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 20)   # expects dtype long as input
        self.reduce = nn.Linear(20, 1)      # dtype: float
        self.example_input_array = torch.tensor([[0, 2, 1], [3, 5, 3]])  # dtype: long

    def forward(self, x):
        return self.reduce(self.embed(x))


@pytest.mark.parametrize(['mode'], [
    pytest.param(ModelSummary.MODE_FULL),
    pytest.param(ModelSummary.MODE_TOP),
])
def test_empty_model_summary_shapes(mode):
    """ Test that the summary works for models that have no submodules. """
    model = EmptyModule()
    summary = model.summarize(mode=mode)
    assert summary.in_sizes == []
    assert summary.out_sizes == []
    assert summary.param_nums == []


@pytest.mark.parametrize(['mode'], [
    pytest.param(ModelSummary.MODE_FULL),
    pytest.param(ModelSummary.MODE_TOP),
])
@pytest.mark.parametrize(['device'], [
    pytest.param(torch.device('cpu')),
    pytest.param(torch.device('cuda', 0)),
    pytest.param(torch.device('cuda', 0)),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_linear_model_summary_shapes(device, mode):
    """ Test that the model summary correctly computes the input- and output shapes. """
    model = UnorderedModel().to(device)
    model.train()
    summary = model.summarize(mode=mode)
    assert summary.in_sizes == [
        [2, 10],    # layer 2
        [2, 7],     # combine
        [2, 3],     # layer 1
        [2, 7],     # relu
        UNKNOWN_SIZE,
    ]
    assert summary.out_sizes == [
        [2, 2],     # layer 2
        [2, 9],     # combine
        [2, 5],     # layer 1
        [2, 7],     # relu
        UNKNOWN_SIZE,
    ]
    assert model.training
    assert model.device == device


def test_mixed_dtype_model_summary():
    """ Test that the model summary works with models that have mixed input- and parameter dtypes. """
    model = MixedDtypeModel()
    summary = model.summarize()
    assert summary.in_sizes == [
        [2, 3],         # embed
        [2, 3, 20],     # reduce
    ]
    assert summary.out_sizes == [
        [2, 3, 20],     # embed
        [2, 3, 1],      # reduce
    ]


@pytest.mark.parametrize(['mode'], [
    pytest.param(ModelSummary.MODE_FULL),
    pytest.param(ModelSummary.MODE_TOP),
])
def test_hooks_removed_after_summarize(mode):
    """ Test that all hooks were properly removed after summary, even ones that were not run. """
    model = UnorderedModel()
    summary = ModelSummary(model, mode=mode)
    # hooks should be removed
    for _, layer in summary.summarize().items():
        handle = layer._hook_handle
        assert handle.id not in handle.hooks_dict_ref()


@pytest.mark.parametrize(['mode'], [
    pytest.param(ModelSummary.MODE_FULL),
    pytest.param(ModelSummary.MODE_TOP),
])
def test_rnn_summary_shapes(mode):
    """ Test that the model summary works for RNNs. """
    model = ParityModuleRNN()

    b = 3
    t = 5
    i = model.rnn.input_size
    h = model.rnn.hidden_size
    o = model.linear_out.out_features

    model.example_input_array = torch.zeros(b, t, 10)

    summary = model.summarize(mode=mode)
    assert summary.in_sizes == [
        [b, t, i],  # rnn
        [b, t, h],  # linear
    ]
    assert summary.out_sizes == [
        [[b, t, h], [[1, b, h], [1, b, h]]],    # rnn
        [b, t, o]                               # linear
    ]


@pytest.mark.parametrize(['mode'], [
    pytest.param(ModelSummary.MODE_FULL),
    pytest.param(ModelSummary.MODE_TOP),
])
def test_summary_parameter_count(mode):
    """ Test that the summary counts the number of parameters in every submodule. """
    model = UnorderedModel()
    summary = model.summarize(mode=mode)
    assert summary.param_nums == [
        model.layer2.weight.numel() + model.layer2.bias.numel(),
        model.combine.weight.numel() + model.combine.bias.numel(),
        model.layer1.weight.numel() + model.layer1.bias.numel(),
        0,  # ReLU
        model.unused.weight.numel() + model.unused.bias.numel(),
    ]


@pytest.mark.parametrize(['mode'], [
    pytest.param(ModelSummary.MODE_FULL),
    pytest.param(ModelSummary.MODE_TOP),
])
def test_summary_layer_types(mode):
    """ Test that the summary displays the layer names correctly. """
    model = UnorderedModel()
    summary = model.summarize(mode=mode)
    assert summary.layer_types == [
        'Linear',
        'Linear',
        'Linear',
        'ReLU',
        'Conv2d',
    ]


@pytest.mark.parametrize(['mode'], [
    pytest.param(ModelSummary.MODE_FULL),
    pytest.param(ModelSummary.MODE_TOP),
])
@pytest.mark.parametrize(['example_input', 'expected_size'], [
    pytest.param([], UNKNOWN_SIZE),
    pytest.param((1, 2, 3), [UNKNOWN_SIZE] * 3),
    pytest.param(torch.tensor(0), UNKNOWN_SIZE),
    pytest.param(dict(tensor=torch.zeros(1, 2, 3)), UNKNOWN_SIZE),
    pytest.param(torch.zeros(2, 3, 4), [2, 3, 4]),
    pytest.param([torch.zeros(2, 3), torch.zeros(4, 5)], [[2, 3], [4, 5]]),
    pytest.param((torch.zeros(2, 3), torch.zeros(4, 5)), [[2, 3], [4, 5]]),
])
def test_example_input_array_types(example_input, expected_size, mode):
    """ Test the types of example inputs supported for display in the summary. """

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

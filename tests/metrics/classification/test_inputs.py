import pytest
import torch

from pytorch_lightning.metrics.utils import to_onehot, select_topk
from pytorch_lightning.metrics.classification.utils import _input_format_classification
from tests.metrics.classification.inputs import (
    Input,
    _binary_inputs,
    _binary_prob_inputs,
    _multiclass_inputs,
    _multiclass_prob_inputs,
    _multidim_multiclass_inputs,
    _multidim_multiclass_prob_inputs,
    _multilabel_inputs,
    _multilabel_prob_inputs,
)
from tests.metrics.utils import NUM_CLASSES, THRESHOLD


#####################################################################
# Test that input transformation works as expected for normal inputs
#####################################################################


def test_binary_inputs():
    inputs = _binary_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0])
    assert mode == "binary"
    assert torch.equal(target, inputs.target[0].unsqueeze(-1))
    assert torch.equal(preds, inputs.preds[0].to(torch.int).unsqueeze(-1))


def test_binary_prob_inputs():
    inputs = _binary_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD)
    assert mode == "binary"
    assert torch.equal(target, inputs.target[0].unsqueeze(-1))
    assert torch.equal(preds, (inputs.preds[0] >= THRESHOLD).to(torch.int).unsqueeze(-1))


def test_multilabel_prob_inputs():
    inputs = _multilabel_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD)
    assert mode == "multi-label"
    assert torch.equal(target, inputs.target[0])
    assert torch.equal(preds, (inputs.preds[0] >= THRESHOLD).to(torch.int))


# Transform preds to floats
def test_multilabel_inputs():
    inputs = _multilabel_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0].to(torch.float), inputs.target[0], THRESHOLD)
    assert mode == "multi-label"
    assert torch.equal(target, inputs.target[0])
    assert torch.equal(preds, inputs.preds[0].to(torch.int))


def test_multiclass_inputs():
    inputs = _multiclass_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD, NUM_CLASSES)
    assert mode == "multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, to_onehot(inputs.preds[0], NUM_CLASSES))


def test_multiclass_prob_inputs():
    inputs = _multiclass_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD)
    assert mode == "multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, select_topk(inputs.preds[0]))


def test_multidim_multiclass_inputs():
    inputs = _multidim_multiclass_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD)
    assert mode == "multi-dim multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, to_onehot(inputs.preds[0], NUM_CLASSES))


def test_multidim_multiclass_prob_inputs():
    inputs = _multidim_multiclass_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD)
    assert mode == "multi-dim multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, select_topk(inputs.preds[0]))


########################################################################
# Test that input transformation works as expected for valid edge cases
########################################################################


def test_binary_as_multiclass():
    inputs = Input(torch.Tensor([0.1, 0.6, 0.2]), torch.Tensor([0, 0, 1]).to(torch.int))
    preds, target, mode = _input_format_classification(inputs.preds, inputs.target, THRESHOLD, 2)

    assert mode == "multi-class"
    assert torch.equal(preds, torch.Tensor([[1, 0], [0, 1], [1, 0]]))
    assert torch.equal(preds, torch.Tensor([[1, 0], [1, 0], [0, 1]]))
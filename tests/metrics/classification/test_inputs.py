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
    _multilabel_multidim_inputs,
    _multilabel_multidim_prob_inputs,
)
from tests.metrics.utils import NUM_CLASSES, THRESHOLD


#####################################################################
# Test that input transformation works as expected for normal inputs
#####################################################################


def test_binary_inputs():
    inputs = _binary_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], is_multiclass=False)
    assert mode == "multi-class"
    assert torch.equal(target, inputs.target[0].unsqueeze(-1))
    assert torch.equal(preds, inputs.preds[0].unsqueeze(-1))


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


def test_multilabel_inputs():
    inputs = _multilabel_inputs
    preds, target, mode = _input_format_classification(
        inputs.preds[0], inputs.target[0], THRESHOLD, is_multiclass=False
    )
    assert mode == "multi-dim multi-class"
    assert torch.equal(target, inputs.target[0])
    assert torch.equal(preds, inputs.preds[0])


def test_multilabel_multidim_prob_inputs():
    inputs = _multilabel_multidim_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD)
    assert mode == "multi-label"
    assert torch.equal(target, inputs.target[0].reshape(inputs.target.shape[1], -1))
    assert torch.equal(preds, (inputs.preds[0] >= THRESHOLD).to(torch.int).reshape(inputs.preds.shape[1], -1))


def test_multilabel_multidim_inputs():
    inputs = _multilabel_multidim_inputs
    preds, target, mode = _input_format_classification(
        inputs.preds[0], inputs.target[0], THRESHOLD, is_multiclass=False
    )
    assert mode == "multi-dim multi-class"
    assert torch.equal(target, inputs.target[0].reshape(inputs.target.shape[1], -1))
    assert torch.equal(preds, inputs.preds[0].reshape(inputs.preds.shape[1], -1))


def test_multiclass_inputs():
    inputs = _multiclass_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], num_classes=NUM_CLASSES)
    assert mode == "multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, to_onehot(inputs.preds[0], NUM_CLASSES))


def test_multiclass_prob_inputs():
    inputs = _multiclass_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0])
    assert mode == "multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, select_topk(inputs.preds[0], 1))


def test_multiclass_prob_inputs_top2():
    inputs = _multiclass_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], top_k=2)
    assert mode == "multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, select_topk(inputs.preds[0], 2))


def test_multidim_multiclass_inputs():
    inputs = _multidim_multiclass_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0])

    assert mode == "multi-dim multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, to_onehot(inputs.preds[0], NUM_CLASSES))


def test_multidim_multiclass_prob_inputs():
    inputs = _multidim_multiclass_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0])

    # Should also work with C dimension in last place
    preds1, target1, mode1 = _input_format_classification(inputs.preds[0].transpose(1, 2), inputs.target[0])

    assert mode == mode1 == "multi-dim multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, select_topk(inputs.preds[0], 1))

    assert torch.equal(target1, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds1, select_topk(inputs.preds[0], 1))


def test_multidim_multiclass_prob_inputs_top2():
    inputs = _multidim_multiclass_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], top_k=2)

    # Should also work with C dimension in last place
    preds1, target1, mode1 = _input_format_classification(inputs.preds[0].transpose(1, 2), inputs.target[0], top_k=2)

    assert mode == mode1 == "multi-dim multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds, select_topk(inputs.preds[0], 2))

    assert torch.equal(target1, to_onehot(inputs.target[0], NUM_CLASSES))
    assert torch.equal(preds1, select_topk(inputs.preds[0], 2))


########################################################################
# Test that input transformation works as expected for valid edge cases
########################################################################


def test_binary_as_multiclass():
    inputs = _binary_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0])

    assert mode == "multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], 2))
    assert torch.equal(preds, to_onehot(inputs.preds[0], 2))


def test_binary_prob_as_multiclass():
    inputs = _binary_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD, is_multiclass=True)

    assert mode == "binary"
    assert torch.equal(preds, to_onehot((inputs.preds[0] >= THRESHOLD).to(torch.int), 2))
    assert torch.equal(target, to_onehot(inputs.target[0], 2))


def test_multilabel_as_multiclass():
    inputs = _multilabel_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], is_multiclass=True)

    assert mode == "multi-dim multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0], 2))
    assert torch.equal(preds, to_onehot(inputs.preds[0], 2))


def test_multilabel_prob_as_multiclass():
    inputs = _multilabel_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD, is_multiclass=True)

    assert mode == "multi-label"
    assert torch.equal(target, to_onehot(inputs.target[0], 2))
    assert torch.equal(preds, to_onehot((inputs.preds[0] >= THRESHOLD).to(torch.int), 2))


def test_multilabel_multidim_as_multiclass():
    inputs = _multilabel_multidim_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], is_multiclass=True)

    assert mode == "multi-dim multi-class"
    assert torch.equal(target, to_onehot(inputs.target[0].reshape(inputs.target.shape[1], -1), 2))
    assert torch.equal(preds, to_onehot(inputs.preds[0].reshape(inputs.preds.shape[1], -1), 2))


def test_multilabel_multidim_prob_as_multiclass():
    inputs = _multilabel_multidim_prob_inputs
    preds, target, mode = _input_format_classification(inputs.preds[0], inputs.target[0], THRESHOLD, is_multiclass=True)

    assert mode == "multi-label"
    assert torch.equal(target, to_onehot(inputs.target[0].reshape(inputs.target.shape[1], -1), 2))
    assert torch.equal(
        preds, to_onehot((inputs.preds[0] >= THRESHOLD).to(torch.int).reshape(inputs.target.shape[1], -1), 2)
    )


def test_multiclass_prob_as_binary():
    inputs = Input(torch.randn(7, 2), torch.randint(high=2, size=(7, 1)))
    preds, target, mode = _input_format_classification(inputs.preds, inputs.target, logits=True, is_multiclass=False)

    assert mode == "multi-class"
    assert torch.equal(target, inputs.target)
    assert torch.equal(preds, select_topk(inputs.preds, 1)[:, [1]])


def test_multidim_multiclass_prob_as_multilabel():
    inputs = Input(torch.randn(7, 2, 3), torch.randint(high=2, size=(7, 3)))
    preds, target, mode = _input_format_classification(inputs.preds, inputs.target, logits=True, is_multiclass=False)

    # Should also work with C dimension in last place
    preds1, target1, mode1 = _input_format_classification(
        inputs.preds.transpose(1, 2), inputs.target, logits=True, is_multiclass=False
    )

    assert mode1 == mode == "multi-dim multi-class"
    assert torch.equal(target, inputs.target)
    assert torch.equal(preds, select_topk(inputs.preds, 1)[:, 1, ...])

    assert torch.equal(target1, inputs.target)
    assert torch.equal(preds1, select_topk(inputs.preds, 1)[:, 1, ...])

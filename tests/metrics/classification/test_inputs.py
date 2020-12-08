import pytest
import torch
from torch import randint, rand

from pytorch_lightning.metrics.utils import to_onehot, select_topk
from pytorch_lightning.metrics.classification.helpers import _input_format_classification
from tests.metrics.classification.inputs import (
    Input,
    _binary_inputs as _bin,
    _binary_prob_inputs as _bin_prob,
    _multiclass_inputs as _mc,
    _multiclass_prob_inputs as _mc_prob,
    _multidim_multiclass_inputs as _mdmc,
    _multidim_multiclass_prob_inputs as _mdmc_prob,
    _multilabel_inputs as _ml,
    _multilabel_prob_inputs as _ml_prob,
    _multilabel_multidim_inputs as _mlmd,
    _multilabel_multidim_prob_inputs as _mlmd_prob,
)
from tests.metrics.utils import NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM, THRESHOLD

torch.manual_seed(42)

# Some additional inputs to test on
_mc_prob_2cls_preds = rand(NUM_BATCHES, BATCH_SIZE, 2)
_mc_prob_2cls_preds /= _mc_prob_2cls_preds.sum(dim=2, keepdim=True)
_mc_prob_2cls = Input(_mc_prob_2cls_preds, randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)))

_mdmc_prob_many_dims_preds = rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM, EXTRA_DIM)
_mdmc_prob_many_dims_preds /= _mdmc_prob_many_dims_preds.sum(dim=2, keepdim=True)
_mdmc_prob_many_dims = Input(
    _mdmc_prob_many_dims_preds,
    randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, EXTRA_DIM)),
)

_mdmc_prob_2cls_preds = rand(NUM_BATCHES, BATCH_SIZE, 2, EXTRA_DIM)
_mdmc_prob_2cls_preds /= _mdmc_prob_2cls_preds.sum(dim=2, keepdim=True)
_mdmc_prob_2cls = Input(_mdmc_prob_2cls_preds, randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)))

# Some utils
T = torch.Tensor


def _idn(x):
    return x


def _usq(x):
    return x.unsqueeze(-1)


def _thrs(x):
    return x >= THRESHOLD


def _rshp1(x):
    return x.reshape(x.shape[0], -1)


def _rshp2(x):
    return x.reshape(x.shape[0], x.shape[1], -1)


def _onehot(x):
    return to_onehot(x, NUM_CLASSES)


def _onehot2(x):
    return to_onehot(x, 2)


def _top1(x):
    return select_topk(x, 1)


def _top2(x):
    return select_topk(x, 2)


# To avoid ugly black line wrapping
def _ml_preds_tr(x):
    return _rshp1(_thrs(x))


def _onehot_rshp1(x):
    return _onehot(_rshp1(x))


def _onehot2_rshp1(x):
    return _onehot2(_rshp1(x))


def _top1_rshp2(x):
    return _top1(_rshp2(x))


def _top2_rshp2(x):
    return _top2(_rshp2(x))


def _probs_to_mc_preds_tr(x):
    return _onehot2(_thrs(x))


def _mlmd_prob_to_mc_preds_tr(x):
    return _onehot2(_rshp1(_thrs(x)))


########################
# Test correct inputs
########################


@pytest.mark.parametrize(
    "inputs, num_classes, is_multiclass, top_k, exp_mode, post_preds, post_target",
    [
        #############################
        # Test usual expected cases
        (_bin, None, False, None, "multi-class", _usq, _usq),
        (_bin, 1, False, None, "multi-class", _usq, _usq),
        (_bin_prob, None, None, None, "binary", lambda x: _usq(_thrs(x)), _usq),
        (_ml_prob, None, None, None, "multi-label", _thrs, _idn),
        (_ml, None, False, None, "multi-dim multi-class", _idn, _idn),
        (_ml_prob, None, None, None, "multi-label", _ml_preds_tr, _rshp1),
        (_mlmd, None, False, None, "multi-dim multi-class", _rshp1, _rshp1),
        (_mc, NUM_CLASSES, None, None, "multi-class", _onehot, _onehot),
        (_mc_prob, None, None, None, "multi-class", _top1, _onehot),
        (_mc_prob, None, None, 2, "multi-class", _top2, _onehot),
        (_mdmc, NUM_CLASSES, None, None, "multi-dim multi-class", _onehot, _onehot),
        (_mdmc_prob, None, None, None, "multi-dim multi-class", _top1_rshp2, _onehot),
        (_mdmc_prob, None, None, 2, "multi-dim multi-class", _top2_rshp2, _onehot),
        (_mdmc_prob_many_dims, None, None, None, "multi-dim multi-class", _top1_rshp2, _onehot_rshp1),
        (_mdmc_prob_many_dims, None, None, 2, "multi-dim multi-class", _top2_rshp2, _onehot_rshp1),
        ###########################
        # Test some special cases
        # Binary as multiclass
        (_bin, None, None, None, "multi-class", _onehot2, _onehot2),
        # Binary probs as multiclass
        (_bin_prob, None, True, None, "binary", _probs_to_mc_preds_tr, _onehot2),
        # Multilabel as multiclass
        (_ml, None, True, None, "multi-dim multi-class", _onehot2, _onehot2),
        # Multilabel probs as multiclass
        (_ml_prob, None, True, None, "multi-label", _probs_to_mc_preds_tr, _onehot2),
        # Multidim multilabel as multiclass
        (_mlmd, None, True, None, "multi-dim multi-class", _onehot2_rshp1, _onehot2_rshp1),
        # Multidim multilabel probs as multiclass
        (_mlmd_prob, None, True, None, "multi-label", _mlmd_prob_to_mc_preds_tr, _onehot2_rshp1),
        # Multiclass prob with 2 classes as binary
        (_mc_prob_2cls, None, False, None, "multi-class", lambda x: _top1(x)[:, [1]], _usq),
        # Multi-dim multi-class with 2 classes as multi-label
        (_mdmc_prob_2cls, None, False, None, "multi-dim multi-class", lambda x: _top1(x)[:, 1], _idn),
    ],
)
def test_usual_cases(inputs, num_classes, is_multiclass, top_k, exp_mode, post_preds, post_target):
    preds_out, target_out, mode = _input_format_classification(
        preds=inputs.preds[0],
        target=inputs.target[0],
        threshold=THRESHOLD,
        num_classes=num_classes,
        is_multiclass=is_multiclass,
        top_k=top_k,
    )

    assert mode == exp_mode
    assert torch.equal(preds_out, post_preds(inputs.preds[0]).int())
    assert torch.equal(target_out, post_target(inputs.target[0]).int())

    # Test that things work when batch_size = 1
    preds_out, target_out, mode = _input_format_classification(
        preds=inputs.preds[0][[0], ...],
        target=inputs.target[0][[0], ...],
        threshold=THRESHOLD,
        num_classes=num_classes,
        is_multiclass=is_multiclass,
        top_k=top_k,
    )

    assert mode == exp_mode
    assert torch.equal(preds_out, post_preds(inputs.preds[0][[0], ...]).int())
    assert torch.equal(target_out, post_target(inputs.target[0][[0], ...]).int())


# Test that threshold is correctly applied
def test_threshold():
    target = T([1, 1, 1]).int()
    preds_probs = T([0.5 - 1e-5, 0.5, 0.5 + 1e-5])

    preds_probs_out, _, _ = _input_format_classification(preds_probs, target, threshold=0.5)

    assert torch.equal(torch.tensor([0, 1, 1], dtype=torch.int), preds_probs_out.squeeze().int())


########################################################################
# Test incorrect inputs
########################################################################


def test_incorrect_threshold():
    with pytest.raises(ValueError):
        _input_format_classification(preds=rand(size=(7,)), target=randint(high=2, size=(7,)), threshold=1.5)


@pytest.mark.parametrize(
    "preds, target, num_classes, is_multiclass",
    [
        # Target not integer
        (randint(high=2, size=(7,)), randint(high=2, size=(7,)).float(), None, None),
        # Target negative
        (randint(high=2, size=(7,)), -randint(high=2, size=(7,)), None, None),
        # Preds negative integers
        (-randint(high=2, size=(7,)), randint(high=2, size=(7,)), None, None),
        # Negative probabilities
        (-rand(size=(7,)), randint(high=2, size=(7,)), None, None),
        # is_multiclass=False and target > 1
        (rand(size=(7,)), randint(low=2, high=4, size=(7,)), None, False),
        # is_multiclass=False and preds integers with > 1
        (randint(low=2, high=4, size=(7,)), randint(high=2, size=(7,)), None, False),
        # Wrong batch size
        (randint(high=2, size=(8,)), randint(high=2, size=(7,)), None, None),
        # Completely wrong shape
        (randint(high=2, size=(7,)), randint(high=2, size=(7, 4)), None, None),
        # Same #dims, different shape
        (randint(high=2, size=(7, 3)), randint(high=2, size=(7, 4)), None, None),
        # Same shape and preds floats, target not binary
        (rand(size=(7, 3)), randint(low=2, high=4, size=(7, 3)), None, None),
        # #dims in preds = 1 + #dims in target, C shape not second or last
        (rand(size=(7, 3, 4, 3)), randint(high=4, size=(7, 3, 3)), None, None),
        # #dims in preds = 1 + #dims in target, preds not float
        (randint(high=2, size=(7, 3, 3, 4)), randint(high=4, size=(7, 3, 3)), None, None),
        # is_multiclass=False, with C dimension > 2
        (_mc_prob.preds[0], randint(high=2, size=(BATCH_SIZE,)), None, False),
        # Probs of multiclass preds do not sum up to 1
        (rand(size=(7, 3, 5)), randint(high=2, size=(7, 5)), None, None),
        # Max target larger or equal to C dimension
        (_mc_prob.preds[0], randint(low=NUM_CLASSES + 1, high=100, size=(BATCH_SIZE,)), None, None),
        # C dimension not equal to num_classes
        (_mc_prob.preds[0], _mc_prob.target[0], NUM_CLASSES + 1, None),
        # Max target larger than num_classes (with #dim preds = 1 + #dims target)
        (_mc_prob.preds[0], randint(low=NUM_CLASSES + 1, high=100, size=(BATCH_SIZE, NUM_CLASSES)), 4, None),
        # Max target larger than num_classes (with #dim preds = #dims target)
        (randint(high=4, size=(7, 3)), randint(low=5, high=7, size=(7, 3)), 4, None),
        # Max preds larger than num_classes (with #dim preds = #dims target)
        (randint(low=5, high=7, size=(7, 3)), randint(high=4, size=(7, 3)), 4, None),
        # Num_classes=1, but is_multiclass not false
        (randint(high=2, size=(7,)), randint(high=2, size=(7,)), 1, None),
        # is_multiclass=False, but implied class dimension (for multi-label, from shape) != num_classes
        (randint(high=2, size=(7, 3, 3)), randint(high=2, size=(7, 3, 3)), 4, False),
        # Multilabel input with implied class dimension != num_classes
        (rand(size=(7, 3, 3)), randint(high=2, size=(7, 3, 3)), 4, False),
        # Multilabel input with is_multiclass=True, but num_classes != 2 (or None)
        (rand(size=(7, 3)), randint(high=2, size=(7, 3)), 4, True),
        # Binary input, num_classes > 2
        (rand(size=(7,)), randint(high=2, size=(7,)), 4, None),
        # Binary input, num_classes == 2 and is_multiclass not True
        (rand(size=(7,)), randint(high=2, size=(7,)), 2, None),
        (rand(size=(7,)), randint(high=2, size=(7,)), 2, False),
        # Binary input, num_classes == 1 and is_multiclass=True
        (rand(size=(7,)), randint(high=2, size=(7,)), 1, True),
    ],
)
def test_incorrect_inputs(preds, target, num_classes, is_multiclass):
    with pytest.raises(ValueError):
        _input_format_classification(
            preds=preds, target=target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass
        )


@pytest.mark.parametrize(
    "preds, target, num_classes, is_multiclass, top_k",
    [
        # Topk set with non (md)mc prob data
        (_bin.preds[0], _bin.target[0], None, None, 2),
        (_bin_prob.preds[0], _bin_prob.target[0], None, None, 2),
        (_mc.preds[0], _mc.target[0], None, None, 2),
        (_ml.preds[0], _ml.target[0], None, None, 2),
        (_mlmd.preds[0], _mlmd.target[0], None, None, 2),
        (_ml_prob.preds[0], _ml_prob.target[0], None, None, 2),
        (_mlmd_prob.preds[0], _mlmd_prob.target[0], None, None, 2),
        (_mdmc.preds[0], _mdmc.target[0], None, None, 2),
        # top_k =2 with 2 classes, is_multiclass=False
        (_mc_prob_2cls.preds[0], _mc_prob_2cls.target[0], None, False, 2),
        # top_k = number of classes (C dimension)
        (_mc_prob.preds[0], _mc_prob.target[0], None, None, NUM_CLASSES),
    ],
)
def test_incorrect_inputs_topk(preds, target, num_classes, is_multiclass, top_k):
    with pytest.raises(ValueError):
        _input_format_classification(
            preds=preds,
            target=target,
            threshold=THRESHOLD,
            num_classes=num_classes,
            is_multiclass=is_multiclass,
            top_k=top_k,
        )

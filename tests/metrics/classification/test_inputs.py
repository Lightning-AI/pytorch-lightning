import pytest
import torch
from torch import randint, rand

from pytorch_lightning.metrics.utils import to_onehot, select_topk
from pytorch_lightning.metrics.classification.utils import _input_format_classification
from tests.metrics.classification.inputs import (
    Input,
    _binary_inputs as _bin,
    _binary_prob_inputs as _bin_prob,
    _multiclass_inputs as _mc,
    _multiclass_prob_inputs as _mc_prob,
    _multidim_multiclass_inputs as _mdmc,
    _multidim_multiclass_prob_inputs as _mdmc_prob,
    _multidim_multiclass_prob_inputs1 as _mdmc_prob1,
    _multilabel_inputs as _ml,
    _multilabel_prob_inputs as _ml_prob,
    _multilabel_multidim_inputs as _mlmd,
    _multilabel_multidim_prob_inputs as _mlmd_prob,
)
from tests.metrics.utils import NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM, THRESHOLD

torch.manual_seed(42)

# Some additional inputs to test on
_mc_prob_2cls = Input(rand(NUM_BATCHES, BATCH_SIZE, 2), randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)))
_mdmc_prob_many_dims = Input(
    rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM, EXTRA_DIM),
    randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, EXTRA_DIM)),
)
_mdmc_prob_many_dims1 = Input(
    rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, EXTRA_DIM, NUM_CLASSES),
    randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, EXTRA_DIM)),
)
_mdmc_prob_2cls = Input(
    rand(NUM_BATCHES, BATCH_SIZE, 2, EXTRA_DIM), randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM))
)
_mdmc_prob_2cls1 = Input(
    rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, 2), randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM))
)

# Some utils
T = torch.Tensor
I = lambda x: x
usq = lambda x: x.unsqueeze(-1)
toint = lambda x: x.int()
thrs = lambda x: x >= THRESHOLD
rshp1 = lambda x: x.reshape(x.shape[0], -1)
rshp2 = lambda x: x.reshape(x.shape[0], x.shape[1], -1)
onehot = lambda x: to_onehot(x, NUM_CLASSES)
onehot2 = lambda x: to_onehot(x, 2)
top1 = lambda x: select_topk(x, 1)
top2 = lambda x: select_topk(x, 2)
mvdim = lambda x: torch.movedim(x, -1, 1)

# To avoid ugly black line wrapping
ml_preds_tr = lambda x: rshp1(toint(thrs(x)))
onehot_rshp1 = lambda x: onehot(rshp1(x))
onehot2_rshp1 = lambda x: onehot2(rshp1(x))
top1_rshp2 = lambda x: top1(rshp2(x))
top2_rshp2 = lambda x: top2(rshp2(x))
mdmc1_top1_tr = lambda x: top1(rshp2(mvdim(x)))
mdmc1_top2_tr = lambda x: top2(rshp2(mvdim(x)))
probs_to_mc_preds_tr = lambda x: toint(onehot2(thrs(x)))
mlmd_prob_to_mc_preds_tr = lambda x: onehot2(rshp1(toint(thrs(x))))
mdmc_prob_to_ml_preds_tr = lambda x: top1(mvdim(x))[:, 1]

########################
# Test correct inputs
########################


@pytest.mark.parametrize(
    "inputs, threshold, logits, num_classes, is_multiclass, top_k, exp_mode, post_preds, post_target",
    [
        #############################
        # Test usual expected cases
        (_bin, THRESHOLD, False, None, False, 1, "multi-class", usq, usq),
        (_bin_prob, THRESHOLD, False, None, None, 1, "binary", lambda x: usq(toint(thrs(x))), usq),
        (_ml_prob, THRESHOLD, False, None, None, 1, "multi-label", lambda x: toint(thrs(x)), I),
        (_ml, THRESHOLD, False, None, False, 1, "multi-dim multi-class", I, I),
        (_ml_prob, THRESHOLD, False, None, None, 1, "multi-label", ml_preds_tr, rshp1),
        (_mlmd, THRESHOLD, False, None, False, 1, "multi-dim multi-class", rshp1, rshp1),
        (_mc, THRESHOLD, False, NUM_CLASSES, None, 1, "multi-class", onehot, onehot),
        (_mc_prob, THRESHOLD, False, None, None, 1, "multi-class", top1, onehot),
        (_mc_prob, THRESHOLD, False, None, None, 2, "multi-class", top2, onehot),
        (_mdmc, THRESHOLD, False, NUM_CLASSES, None, 1, "multi-dim multi-class", onehot, onehot),
        (_mdmc_prob, THRESHOLD, False, None, None, 1, "multi-dim multi-class", top1_rshp2, onehot),
        (_mdmc_prob, THRESHOLD, False, None, None, 2, "multi-dim multi-class", top2_rshp2, onehot),
        (_mdmc_prob_many_dims, THRESHOLD, False, None, None, 1, "multi-dim multi-class", top1_rshp2, onehot_rshp1),
        (_mdmc_prob_many_dims, THRESHOLD, False, None, None, 2, "multi-dim multi-class", top2_rshp2, onehot_rshp1),
        # Test with C dim in last place
        (_mdmc_prob1, THRESHOLD, False, None, None, 1, "multi-dim multi-class", mdmc1_top1_tr, onehot),
        (_mdmc_prob1, THRESHOLD, False, None, None, 2, "multi-dim multi-class", mdmc1_top2_tr, onehot),
        (_mdmc_prob_many_dims1, THRESHOLD, False, None, None, 1, "multi-dim multi-class", mdmc1_top1_tr, onehot_rshp1),
        (_mdmc_prob_many_dims1, THRESHOLD, False, None, None, 2, "multi-dim multi-class", mdmc1_top2_tr, onehot_rshp1),
        ###########################
        # Test some special cases
        # Binary as multiclass
        (_bin, THRESHOLD, False, None, None, 1, "multi-class", onehot2, onehot2),
        # Binary probs as multiclass
        (_bin_prob, THRESHOLD, False, None, True, 1, "binary", probs_to_mc_preds_tr, onehot2),
        # Multilabel as multiclass
        (_ml, THRESHOLD, False, None, True, 1, "multi-dim multi-class", onehot2, onehot2),
        # Multilabel probs as multiclass
        (_ml_prob, THRESHOLD, False, None, True, 1, "multi-label", probs_to_mc_preds_tr, onehot2),
        # Multidim multilabel as multiclass
        (_mlmd, THRESHOLD, False, None, True, 1, "multi-dim multi-class", onehot2_rshp1, onehot2_rshp1),
        # Multidim multilabel probs as multiclass
        (_mlmd_prob, THRESHOLD, False, None, True, 1, "multi-label", mlmd_prob_to_mc_preds_tr, onehot2_rshp1),
        # Multiclass prob with 2 classes as binary
        (_mc_prob_2cls, THRESHOLD, False, None, False, 1, "multi-class", lambda x: top1(x)[:, [1]], usq),
        # Multi-dim multi-class with 2 classes as multi-label
        (_mdmc_prob_2cls, THRESHOLD, False, None, False, 1, "multi-dim multi-class", lambda x: top1(x)[:, 1], I),
        (_mdmc_prob_2cls1, THRESHOLD, False, None, False, 1, "multi-dim multi-class", mdmc_prob_to_ml_preds_tr, I),
    ],
)
def test_usual_cases(inputs, threshold, logits, num_classes, is_multiclass, top_k, exp_mode, post_preds, post_target):
    preds_out, target_out, mode = _input_format_classification(
        preds=inputs.preds[0],
        target=inputs.target[0],
        threshold=threshold,
        logits=logits,
        num_classes=num_classes,
        is_multiclass=is_multiclass,
        top_k=top_k,
    )

    assert mode == exp_mode
    assert torch.equal(preds_out, post_preds(inputs.preds[0]))
    assert torch.equal(target_out, post_target(inputs.target[0]))

    # Test that things work when batch_size = 1
    preds_out, target_out, mode = _input_format_classification(
        preds=inputs.preds[0][[0], ...],
        target=inputs.target[0][[0], ...],
        threshold=threshold,
        logits=logits,
        num_classes=num_classes,
        is_multiclass=is_multiclass,
        top_k=top_k,
    )

    assert mode == exp_mode
    assert torch.equal(preds_out, post_preds(inputs.preds[0][[0], ...]))
    assert torch.equal(target_out, post_target(inputs.target[0][[0], ...]))


# Test that threshold is correctly transformed in logit cases
def test_logit_threshold():
    target = T([1, 1, 1]).int()
    preds_logit = T([-1e-5, 0, 1e-5])
    preds_probs = T([0.5 - 1e-5, 0.5, 0.5 + 1e-5])

    preds_logit_out, _, _ = _input_format_classification(preds_logit, target, threshold=0.5, logits=True)
    preds_probs_out, _, _ = _input_format_classification(preds_probs, target, threshold=0.5, logits=False)

    assert torch.equal(preds_logit_out, preds_probs_out)


########################################################################
# Test incorrect inputs
########################################################################


@pytest.mark.parametrize(
    "preds, target, threshold, logits, num_classes, is_multiclass, top_k",
    [
        # Target not integer
        (randint(high=2, size=(7,)), randint(high=2, size=(7,)).float(), 0.5, False, None, None, 1),
        # Target negative
        (randint(high=2, size=(7,)), -randint(high=2, size=(7,)), 0.5, False, None, None, 1),
        # Preds negative integers
        (-randint(high=2, size=(7,)), randint(high=2, size=(7,)), 0.5, False, None, None, 1),
        # Negative probabilities
        (-rand(size=(7,)), randint(high=2, size=(7,)), 0.5, False, None, None, 1),
        # Threshold outside of [0,1]
        (rand(size=(7,)), randint(high=2, size=(7,)), 1.5, False, None, None, 1),
        # is_multiclass=False and target > 1
        (rand(size=(7,)), randint(low=2, high=4, size=(7,)), 0.5, False, None, False, 1),
        # is_multiclass=False and preds integers with > 1
        (randint(low=2, high=4, size=(7,)), randint(high=2, size=(7,)), 0.5, False, None, False, 1),
        # Wrong batch size
        (randint(high=2, size=(8,)), randint(high=2, size=(7,)), 0.5, False, None, None, 1),
        # Completely wrong shape
        (randint(high=2, size=(7,)), randint(high=2, size=(7, 4)), 0.5, False, None, None, 1),
        # Same #dims, different shape
        (randint(high=2, size=(7, 3)), randint(high=2, size=(7, 4)), 0.5, False, None, None, 1),
        # Same shape and preds floats, target not binary
        (rand(size=(7, 3)), randint(low=2, high=4, size=(7, 3)), 0.5, False, None, None, 1),
        # #dims in preds = 1 + #dims in target, C shape not second or last
        (rand(size=(7, 3, 4, 3)), randint(high=4, size=(7, 3, 3)), 0.5, False, None, None, 1),
        # #dims in preds = 1 + #dims in target, preds not float
        (randint(high=2, size=(7, 3, 3, 4)), randint(high=4, size=(7, 3, 3)), 0.5, False, None, None, 1),
        # is_multiclass=False, with C dimension > 2
        (rand(size=(7, 3, 5)), randint(high=2, size=(7, 5)), 0.5, False, None, False, 1),
        # Max target larger or equal to C dimension
        (rand(size=(7, 3)), randint(low=4, high=6, size=(7,)), 0.5, False, None, None, 1),
        # C dimension not equal to num_classes
        (rand(size=(7, 3, 4)), randint(high=4, size=(7, 3)), 0.5, False, 7, None, 1),
        # Max target larger than num_classes (with #dim preds = 1 + #dims target)
        (rand(size=(7, 3, 4)), randint(low=5, high=7, size=(7, 3)), 0.5, False, 4, None, 1),
        # Max target larger than num_classes (with #dim preds = #dims target)
        (randint(high=4, size=(7, 3)), randint(low=5, high=7, size=(7, 3)), 0.5, False, 4, None, 1),
        # Max preds larger than num_classes (with #dim preds = #dims target)
        (randint(low=5, high=7, size=(7, 3)), randint(high=4, size=(7, 3)), 0.5, False, 4, None, 1),
        # Num_classes=1, but is_multiclass not false
        (randint(high=2, size=(7,)), randint(high=2, size=(7,)), 0.5, False, 1, None, 1),
        # is_multiclass=False, but implied class dimension (for multi-label, from shape) != num_classes
        (randint(high=2, size=(7, 3, 3)), randint(high=2, size=(7, 3, 3)), 0.5, False, 4, False, 1),
        # Multilabel input with implied class dimension != num_classes
        (rand(size=(7, 3, 3)), randint(high=2, size=(7, 3, 3)), 0.5, False, 4, False, 1),
        # Binary input, num_classes > 2
        (rand(size=(7,)), randint(high=2, size=(7,)), 0.5, False, 4, None, 1),
        # Binary input, num_classes == 2 and is_multiclass not True
        (rand(size=(7,)), randint(high=2, size=(7,)), 0.5, False, 2, None, 1),
        (rand(size=(7,)), randint(high=2, size=(7,)), 0.5, False, 2, False, 1),
        # Binary input, num_classes == 1 and is_multiclass=True
        (rand(size=(7,)), randint(high=2, size=(7,)), 0.5, False, 1, True, 1),
        # Topk > 1 with non (md)mc prob data
        (_bin.preds[0], _bin.target[0], 0.5, False, None, None, 2),
        (_bin_prob.preds[0], _bin_prob.target[0], 0.5, False, None, None, 2),
        (_mc.preds[0], _mc.target[0], 0.5, False, None, None, 2),
        (_ml.preds[0], _ml.target[0], 0.5, False, None, None, 2),
        (_mlmd.preds[0], _mlmd.target[0], 0.5, False, None, None, 2),
        (_ml_prob.preds[0], _ml_prob.target[0], 0.5, False, None, None, 2),
        (_mlmd_prob.preds[0], _mlmd_prob.target[0], 0.5, False, None, None, 2),
        (_mdmc.preds[0], _mdmc.target[0], 0.5, False, None, None, 2),
    ],
)
def test_incorrect_inputs(preds, target, threshold, logits, num_classes, is_multiclass, top_k):
    with pytest.raises(ValueError):
        _input_format_classification(
            preds=preds,
            target=target,
            threshold=threshold,
            logits=logits,
            num_classes=num_classes,
            is_multiclass=is_multiclass,
            top_k=top_k
        )

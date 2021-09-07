import pickle
from copy import deepcopy

import pytest
import torch

from pytorch_lightning.loops.closure import ClosureResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_closure_result_deepcopy():
    closure_loss = torch.tensor(123.45)
    hiddens = torch.tensor(321.12, requires_grad=True)
    result = ClosureResult(closure_loss, hiddens)
    assert not result.hiddens.requires_grad

    assert closure_loss.data_ptr() == result.closure_loss.data_ptr()
    # the `loss` is cloned so the storage is different
    assert closure_loss.data_ptr() != result.loss.data_ptr()

    copy = deepcopy(result)
    assert result.loss == copy.loss
    assert copy.closure_loss is None
    assert copy.hiddens is None

    assert id(result.loss) != id(copy.loss)
    assert result.loss.data_ptr() != copy.loss.data_ptr()

    assert copy == pickle.loads(pickle.dumps(result))


def test_closure_result_raises():
    with pytest.raises(MisconfigurationException, match="If `hiddens` are returned .* the loss cannot be `None`"):
        ClosureResult(None, "something")


def test_closure_result_apply_accumulation():
    closure_loss = torch.tensor(25.0)
    result = ClosureResult(closure_loss, None)
    assert result.loss == 25
    result.apply_accumulation(5)
    assert result.loss == 5

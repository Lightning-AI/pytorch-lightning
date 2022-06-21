import math

import pytest
import torch
import torch.nn as nn

from pytorch_lightning.utilities.finite_checks import detect_nan_parameters


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_detect_nan_parameters(value):
    model = nn.Linear(2, 3)

    detect_nan_parameters(model)

    nn.init.constant_(model.bias, value)
    assert not torch.isfinite(model.bias).all()

    with pytest.raises(ValueError, match=r".*Detected nan and/or inf values in `bias`.*"):
        detect_nan_parameters(model)

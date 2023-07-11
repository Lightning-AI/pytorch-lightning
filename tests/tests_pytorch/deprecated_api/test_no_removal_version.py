import pytest
import torch.nn

import lightning.fabric
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from tests_pytorch.helpers.runif import RunIf


def test_ddp_is_distributed():
    strategy = DDPStrategy()
    with pytest.deprecated_call(match="is deprecated"):
        _ = strategy.is_distributed


@RunIf(min_torch="1.12")
def test_fsdp_activation_checkpointing(monkeypatch):
    with pytest.raises(ValueError, match="cannot set both `activation_checkpointing"):
        FSDPStrategy(activation_checkpointing=torch.nn.Linear, activation_checkpointing_policy=lambda *_: True)

    monkeypatch.setattr(lightning.fabric.strategies.fsdp, "_TORCH_GREATER_EQUAL_2_1", True)
    with pytest.deprecated_call(match=r"use `FSDPStrategy\(activation_checkpointing_policy"):
        FSDPStrategy(activation_checkpointing=torch.nn.Linear)

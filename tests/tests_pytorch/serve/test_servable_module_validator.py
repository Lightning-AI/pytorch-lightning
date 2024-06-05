from typing import Dict

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.serve.servable_module_validator import ServableModule, ServableModuleValidator
from torch import Tensor


class ServableBoringModel(BoringModel, ServableModule):
    def configure_payload(self):
        return {"body": {"x": list(range(32))}}

    def configure_serialization(self):
        def deserialize(x):
            return torch.tensor(x, dtype=torch.float)

        def serialize(x):
            return x.tolist()

        return {"x": deserialize}, {"output": serialize}

    def serve_step(self, x: Tensor) -> Dict[str, Tensor]:
        assert torch.equal(x, torch.arange(32, dtype=torch.float))
        return {"output": torch.tensor([0, 1])}

    def configure_response(self):
        return {"output": [0, 1]}


@pytest.mark.xfail(strict=False, reason="test is too flaky in CI")  # todo
def test_servable_module_validator():
    model = ServableBoringModel()
    callback = ServableModuleValidator()
    callback.on_train_start(Trainer(accelerator="cpu"), model)


@pytest.mark.flaky(reruns=3)
def test_servable_module_validator_with_trainer(tmp_path, mps_count_0):
    callback = ServableModuleValidator()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        callbacks=[callback],
        strategy="ddp_spawn",
        devices=2,
    )
    trainer.fit(ServableBoringModel())

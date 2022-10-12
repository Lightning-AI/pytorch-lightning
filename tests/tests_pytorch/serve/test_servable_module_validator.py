from typing import Dict

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.serve.servable_module_validator import ServableModule, ServableModuleValidator


class ServableBoringModel(BoringModel, ServableModule):
    def configure_payload(self):
        return {"body": {"x": list(range(32))}}

    def configure_serialization(self):
        def deserialize(x):
            return torch.tensor(x, dtype=torch.float)

        def serialize(x):
            return x.tolist()

        return {"x": deserialize}, {"output": serialize}

    def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert torch.equal(x, torch.arange(32, dtype=torch.float))
        return {"output": torch.tensor([0, 1])}

    def configure_response(self):
        return {"output": [0, 1]}


def test_servable_module_validator():
    model = ServableBoringModel()
    callback = ServableModuleValidator()
    callback.on_train_start(Trainer(), model)


@pytest.mark.flaky(reruns=3)
def test_servable_module_validator_with_trainer(tmpdir):
    callback = ServableModuleValidator()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        callbacks=[callback],
        strategy="ddp_spawn",
        devices=2,
    )
    trainer.fit(ServableBoringModel())

from typing import Dict

import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.serve.servable_module_validator import ServableModule, ServableModuleValidator


class ServableBoringModel(BoringModel, ServableModule):
    def configure_payload(self) -> ...:
        return {"body": {"x": list(range(32))}}

    def configure_serialization(self):
        def deserialize(x):
            return torch.tensor(x, dtype=torch.float)

        def serialize(x):
            return x.tolist()

        return {"x": deserialize}, {"output": serialize}

    def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert torch.equal(x, torch.arange(32).float())
        return {"output": torch.tensor([0, 1])}


def test_servable_module_validator():
    seed_everything(42)
    model = ServableBoringModel()
    callback = ServableModuleValidator()
    callback.on_train_start(Trainer(), model)
    assert callback.resp.json() == {"output": [0, 1]}


def test_servable_module_validator_with_trainer():
    seed_everything(42)
    callback = ServableModuleValidator()
    trainer = Trainer(max_epochs=1, limit_train_batches=2, limit_val_batches=0, callbacks=[callback])
    trainer.fit(ServableBoringModel())
    assert callback.resp.json() == {"output": [0, 1]}

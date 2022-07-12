from typing import Dict

import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks.sanity_serving import SanityServing, ServableModule
from pytorch_lightning.demos.boring_classes import BoringModel


class ServableBoringModel(BoringModel, ServableModule):
    def configure_payload(self) -> ...:
        return {"body": {"x": list(range(32))}}

    def configure_inputs_outputs(self):
        class Tensor:
            @staticmethod
            def deserialize(x):
                return torch.tensor(x).float()

            @staticmethod
            def serialize(x):
                return x.numpy().tolist()

        return ({"x": Tensor.deserialize}, {"output": Tensor.serialize})

    def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"output": self.forward(x)}


def test_sanity_serving():
    seed_everything(42)
    model = ServableBoringModel()
    callback = SanityServing()
    callback.on_train_start(None, model)
    assert callback.resp.json() == {"output": [-7.451034069061279, 1.635885238647461]}


def test_sanity_serving_trainer():
    seed_everything(42)
    callback = SanityServing()
    trainer = Trainer(max_epochs=1, limit_train_batches=2, limit_val_batches=0, callbacks=[callback])
    trainer.fit(ServableBoringModel())
    assert callback.resp.json() == {"output": [-7.451034069061279, 1.635885238647461]}

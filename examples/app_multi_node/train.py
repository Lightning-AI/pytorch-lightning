from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel

if __name__ == "__main__":
    model = BoringModel()
    trainer = Trainer(max_epochs=1)
    trainer.fit(model)

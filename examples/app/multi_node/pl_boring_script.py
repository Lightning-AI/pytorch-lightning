import lightning as L
from lightning.pytorch.demos.boring_classes import BoringModel

if __name__ == "__main__":
    model = BoringModel()
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model)

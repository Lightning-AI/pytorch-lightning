from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

if __name__ == "__main__":
    model = BoringModel()
    trainer = Trainer(max_epochs=1, accelerator="cpu", devices=2, strategy="ddp")
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)

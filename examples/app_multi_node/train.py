from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

if __name__ == "__main__":
    model = BoringModel()
    trainer = Trainer(max_epochs=1, strategy="ddp")
    print("Strategy", trainer.strategy.__dict__)
    trainer.fit(model)

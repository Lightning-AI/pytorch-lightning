import sys

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel


class CustomException(Exception):
    pass


class TestModel(BoringModel):
    def training_step(self, batch, batch_idx):
        if batch_idx == 1 and self.trainer.is_global_zero:
            # rank 0: raises an exception
            # rank 1: continues training but will hang on the next barrier in the training loop
            raise CustomException
        return super().training_step(batch, batch_idx)


model = TestModel()

trainer = Trainer(
    default_root_dir=".", max_epochs=1, limit_train_batches=5, num_sanity_val_steps=0, gpus=2, accelerator="ddp"
)

try:
    # simulate random failure in training_step on rank 0
    trainer.fit(model)
except Exception as e:
    assert "CustomException" in str(e)

# sys.exit(0)

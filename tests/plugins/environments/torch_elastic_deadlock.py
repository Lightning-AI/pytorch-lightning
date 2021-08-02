import os
import sys
from contextlib import suppress

from pytorch_lightning import Trainer
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.exceptions import DeadlockDetectedException
from tests.helpers.boring_model import BoringModel

if os.getenv("PL_RUNNING_SPECIAL_TESTS", "0") == "1":

    class CustomException(Exception):
        pass

    class Model(BoringModel):
        def training_step(self, batch, batch_idx):
            if batch_idx == 1 and self.trainer.is_global_zero:
                # rank 0: raises an exception
                # rank 1: continues training but will hang on the next barrier in the training loop
                raise CustomException
            return super().training_step(batch, batch_idx)

    model = Model()

    trainer = Trainer(
        default_root_dir=".", max_epochs=1, limit_train_batches=5, num_sanity_val_steps=0, gpus=2, accelerator="ddp"
    )
    assert isinstance(trainer.training_type_plugin, DDPPlugin)

    with suppress(DeadlockDetectedException):
        # simulate random failure in training_step on rank 0
        trainer.fit(model)

    # used to capture success from this script in the CI.
    print("SUCCEEDED")

    sys.exit(0)

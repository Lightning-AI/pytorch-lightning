# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.exceptions import DeadlockDetectedException


class CustomException(Exception):
    pass


class Model(BoringModel):
    def training_step(self, batch, batch_idx):
        if batch_idx == 1 and self.trainer.is_global_zero:
            # rank 0: raises an exception
            # rank 1: continues training but will hang on the next barrier in the training loop
            raise CustomException
        return super().training_step(batch, batch_idx)


def main():
    model = Model()

    trainer = Trainer(
        default_root_dir=".",
        max_epochs=1,
        limit_train_batches=5,
        num_sanity_val_steps=0,
        accelerator="cuda",
        devices=2,
        strategy="ddp",
    )
    assert isinstance(trainer.strategy, DDPStrategy)

    exception = None
    try:
        trainer.fit(model)
    except DeadlockDetectedException as e:
        exception = e

    # only rank 0 should reach this code
    # rank 1 (which hangs) got killed by rank 0 as part of the process reconcilliation
    assert exception is not None
    assert trainer.global_rank == 0
    assert str(exception) == "DeadLock detected from rank: 0"


if __name__ == "__main__":
    os.environ["PL_RECONCILE_PROCESS"] = "1"
    main()

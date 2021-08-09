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
import functools
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()


class ExternalLoop(Loop):
    """This Loop is meant wrap trainer calls"""

    def __init__(self):
        super().__init__()
        warning_cache.warn("The ExternalLoop API is a `pre-alpha release` and breaking API changes are expected.")
        self.create_trainer = self._wrap_trainer_wrapper(self.create_trainer)
        self._has_setup = False
        self._restore_external_loop = True

    def _wrap_trainer_wrapper(self, create_trainer: Callable) -> Callable:
        @functools.wraps(create_trainer)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            trainer = create_trainer(*args, trainer_kwargs=self.trainer_kwargs, **kwargs)
            if not isinstance(trainer, pl.Trainer):
                raise MisconfigurationException("The `create_trainer` hook should return a Trainer")
            self.trainer = trainer
            self.trainer.external_loop = self

            self.trainer.accelerator.connect(self.__lightning_module)

            # links data to the trainer
            self.trainer.data_connector.attach_data(
                self.trainer.lightning_module,
                train_dataloaders=self.__train_dataloader,
                val_dataloaders=self.__val_dataloaders,
                test_dataloaders=self.__test_dataloaders,
                predict_dataloaders=self.__predict_dataloaders,
                datamodule=self.__datamodule,
            )

            # attach model to the training type plugin
            self.trainer.data_connector.prepare_data()

            self.trainer.checkpoint_connector.resume_start()
            self.trainer.checkpoint_connector.restore_loops(restore_external_loop=self._restore_external_loop)
            return trainer

        return wrapped_func

    def connect_trainer(self, **trainer_kwargs: Dict[str, Any]) -> None:
        self.trainer_kwargs = trainer_kwargs

    def create_trainer(self, *args, trainer_kwargs: Dict[str, Any] = {}, **kwargs) -> "pl.Trainer":
        trainer_kwargs.update(kwargs)
        return pl.Trainer(*args, **trainer_kwargs)

    def run(
        self,
        model: "pl.LightningModule",
        train_dataloader=None,
        val_dataloaders=None,
        test_dataloaders=None,
        predict_dataloaders=None,
        datamodule=None,
    ):

        self.__lightning_module = model
        self.__train_dataloader = train_dataloader
        self.__val_dataloaders = val_dataloaders
        self.__test_dataloaders = test_dataloaders
        self.__predict_dataloaders = predict_dataloaders
        self.__datamodule = datamodule

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloader, pl.LightningDataModule):
            datamodule = train_dataloader
            train_dataloader = None

        if train_dataloader is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `loop.run(dataloaders=..., datamodule=...)`")

        if model is None:
            raise MisconfigurationException("`model` must be provided to `loop.run()`")

        if self._trainer is None:
            self.create_trainer()
            self._restore_external_loop = False

        return super().run()

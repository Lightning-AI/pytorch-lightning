# Copyright Lightning AI team.
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

from torch import Tensor
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def log_hyperparams(trainer) -> None:
    if not trainer.loggers:
        return
    # log hyper-parameters
    hparams_initial = None

    # save exp to get started (this is where the first experiment logs are written)
    datamodule_log_hyperparams = trainer.datamodule._log_hyperparams if trainer.datamodule is not None else False

    if trainer.lightning_module._log_hyperparams and datamodule_log_hyperparams:
        datamodule_hparams = trainer.datamodule.hparams_initial
        lightning_hparams = trainer.lightning_module.hparams_initial
        inconsistent_keys = []
        for key in lightning_hparams.keys() & datamodule_hparams.keys():
            lm_val, dm_val = lightning_hparams[key], datamodule_hparams[key]
            if type(lm_val) != type(dm_val):
                inconsistent_keys.append(key)
            elif isinstance(lm_val, Tensor) and id(lm_val) != id(dm_val):
                inconsistent_keys.append(key)
            elif lm_val != dm_val:
                inconsistent_keys.append(key)
        if inconsistent_keys:
            raise MisconfigurationException(
                f"Error while merging hparams: the keys {inconsistent_keys} are present "
                "in both the LightningModule's and LightningDataModule's hparams "
                "but have different values."
            )
        hparams_initial = {**lightning_hparams, **datamodule_hparams}
    elif trainer.lightning_module._log_hyperparams:
        hparams_initial = trainer.lightning_module.hparams_initial
    elif datamodule_log_hyperparams:
        hparams_initial = trainer.datamodule.hparams_initial

    for logger in trainer.loggers:
        if hparams_initial is not None:
            logger.log_hyperparams(hparams_initial)
        logger.log_graph(trainer.lightning_module)
        logger.save()


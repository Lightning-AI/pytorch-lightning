from pytorch_lightning.trainer.trainer_conf import PLConf
import pl_examples.hydra_examples.conf.optimizer
import pl_examples.hydra_examples.conf.scheduler
import hydra
from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf
from omegaconf import MISSING
from typing import Any, List
from dataclasses import dataclass
from pytorch_lightning import Callback

cs = ConfigStore.instance()

# Sample callback definition used in hydra yaml config
class MyPrintingCallback(Callback):
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("trainer is init now")

    def on_train_end(self, trainer, pl_module):
        print("do something when training ends")


# Sample callback definition with param used in hydra yaml config
class MessageCallback(Callback):
    def __init__(self, iter_num):
        self.iter_num = iter_num

    def on_batch_start(self, trainer, pl_module):
        if trainer.batch_idx == self.iter_num:
            print(f"Iteration {self.iter_num}")


"""
Top Level used config which can be extended by a user.
For use in Pytorch Lightning we can extend the PLConfig
dataclass which has all the trainer settings. For further 
config with type safety, we can extend this class and
add in other config groups. 
"""


@dataclass
class UserConf(PLConf):
    defaults: List[Any] = MISSING
    data: Any = MISSING
    model: Any = MISSING
    scheduler: ObjectConf = MISSING
    opt: ObjectConf = MISSING
    


# Stored as config node, for top level config used for type checking.
cs.store(name="config", node=UserConf)

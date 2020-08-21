"""
HyperTuner n worker searcher
"""
from abc import ABC

from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.hypertuner.utils import check_call_order


class HyperTunerNworkerSearchMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    trainer: Trainer
    model: LightningModule

    @check_call_order
    def n_worker_search(self):
        pass

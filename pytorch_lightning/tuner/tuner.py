

from pytorch_lightning.trainer.training_tricks import TunerLRFinderMixin
from pytorch_lightning.trainer.lr_finder import TunerBatchScalerMixin

class Tuner(
        TunerLRFinderMixin,
        TunerBatchScalerMixin):
    
    def __init__(self, trainer):
        pass
    
    def optimize(self, model):
        pass
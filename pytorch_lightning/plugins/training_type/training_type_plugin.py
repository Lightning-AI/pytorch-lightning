from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.utilities import rank_zero_deprecation

rank_zero_deprecation(
    "TrainingTypePlugin was renamed to Strategy. Import `Strategy` from `pytorch_lightning.strategies.strategy`"
)

TrainingTypePlugin = Strategy

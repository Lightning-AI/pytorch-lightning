from pytorch_lightning.accelerators.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import AMPType


class MixedPrecisionPlugin(PrecisionPlugin):
    EPSILON = 1e-5
    backend: AMPType
    precision = "mixed"
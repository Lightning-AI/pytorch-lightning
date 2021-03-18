from typing import TYPE_CHECKING

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if TYPE_CHECKING:
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.trainer.trainer import Trainer


class CPUAccelerator(Accelerator):

    def setup(self, trainer: 'Trainer', model: 'LightningModule') -> None:
        """
        Raises:
            MisconfigurationException:
                If AMP is used with CPU, or if the selected device is not CPU.
        """
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            raise MisconfigurationException("amp + cpu is not supported. Please use a GPU option")

        if "cpu" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be CPU, got {self.root_device} instead")

        return super().setup(trainer, model)

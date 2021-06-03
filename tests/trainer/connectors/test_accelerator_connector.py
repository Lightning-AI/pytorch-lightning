import torch

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.plugins import DDPPlugin, PrecisionPlugin, SingleDevicePlugin
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector


def test_accelerator_training_type_plugin():
    """Test that training_type_plugin is pulled from accelearator"""

    # check that this works for different types of plugins to ensure
    # there are no dependencies on TrainingTypePlugin class refinements

    precision_plugin = PrecisionPlugin()
    singledev_plugin = SingleDevicePlugin(torch.device('cpu'))
    accelerator = Accelerator(
        precision_plugin=precision_plugin,
        training_type_plugin=singledev_plugin,
    )
    trainer = Trainer(accelerator=accelerator)
    assert trainer.accelerator_connector.training_type_plugin is singledev_plugin

    precision_plugin = PrecisionPlugin()
    ddp_plugin = DDPPlugin()
    accelerator = Accelerator(
        precision_plugin=precision_plugin,
        training_type_plugin=ddp_plugin,
    )
    trainer = Trainer(accelerator=accelerator)
    assert trainer.accelerator_connector.training_type_plugin is ddp_plugin

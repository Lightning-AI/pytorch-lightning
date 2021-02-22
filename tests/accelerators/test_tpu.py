from unittest.mock import Mock

import pytest

from pytorch_lightning.accelerators import TPUAccelerator
from pytorch_lightning.plugins import SingleTPUPlugin, DDPPlugin, PrecisionPlugin
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_unsupported_precision_plugins():
    """ Test error messages are raised for unsupported precision plugins with TPU. """
    trainer = Mock()
    model = Mock()
    accelerator = TPUAccelerator(
        training_type_plugin=SingleTPUPlugin(device=Mock()),
        precision_plugin=MixedPrecisionPlugin(),
    )
    with pytest.raises(MisconfigurationException, match=r"amp \+ tpu is not supported."):
        accelerator.setup(trainer=trainer, model=model)


def test_unsupported_training_type_plugins():
    """ Test error messages are raised for unsupported training type with TPU. """
    trainer = Mock()
    model = Mock()
    accelerator = TPUAccelerator(
        training_type_plugin=DDPPlugin(),
        precision_plugin=PrecisionPlugin(),
    )
    with pytest.raises(MisconfigurationException, match="TPUs only support a single tpu core or tpu spawn training"):
        accelerator.setup(trainer=trainer, model=model)

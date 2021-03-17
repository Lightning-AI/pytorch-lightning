import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from tests.helpers import BoringModel


class CustomParallelPlugin(DDPPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set to None so it will be overwritten by the accelerator connector.
        self.sync_batchnorm = None

@pytest.mark.skipif(torch.cuda.is_available(), reason="test doesn't requires GPU machine")
def test_sync_batchnorm_set(tmpdir):
    """Tests if sync_batchnorm is automatically set for custom plugin."""
    model = BoringModel()
    plugin = CustomParallelPlugin()
    assert plugin.sync_batchnorm is None
    trainer = Trainer(
        max_epochs=1,
        plugins=[plugin],
        default_root_dir=tmpdir,
        sync_batchnorm=True,
    )
    trainer.fit(model)
    assert plugin.sync_batchnorm is True

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from tests.helpers import BoringModel


class CustomParallelPlugin(DDPPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def test_sync_batchnorm_set(tmpdir):
    model = BoringModel()
    plugin = CustomParallelPlugin()
    assert plugin.sync_batchnorm is False
    trainer = Trainer(
        max_epochs=1,
        plugins=[plugin],
        default_root_dir=tmpdir,
        sync_batchnorm=True,
    )
    trainer.fit(model)
    assert plugin.sync_batchnorm is True

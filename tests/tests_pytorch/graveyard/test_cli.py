import pytest


def test_lightningCLI_old_module_removal():
    from pytorch_lightning.utilities.cli import LightningCLI

    with pytest.raises(NotImplementedError, match=r"LightningCLI.*no longer supported as of v1.9"):
        LightningCLI()

    from pytorch_lightning.utilities.cli import SaveConfigCallback

    with pytest.raises(NotImplementedError, match=r"SaveConfigCallback.*no longer supported as of v1.9"):
        SaveConfigCallback()

    from pytorch_lightning.utilities.cli import LightningArgumentParser

    with pytest.raises(NotImplementedError, match=r"LightningArgumentParser.*no longer supported as of v1.9"):
        LightningArgumentParser()

    from pytorch_lightning.utilities.cli import instantiate_class

    with pytest.raises(NotImplementedError, match=r"instantiate_class.*no longer supported as of v1.9"):
        instantiate_class()

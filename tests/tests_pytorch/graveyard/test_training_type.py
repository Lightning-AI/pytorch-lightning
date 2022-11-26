from importlib import import_module

import pytest

from pytorch_lightning import Trainer


def test_removed_training_type_plugin_property():
    trainer = Trainer()
    with pytest.raises(AttributeError, match="training_type_plugin`.*no longer accessible as of v1.8"):
        trainer.training_type_plugin


@pytest.mark.parametrize(
    "name",
    (
        "DDPPlugin",
        "DDP2Plugin",
        "DDPSpawnPlugin",
        "DeepSpeedPlugin",
        "DataParallelPlugin",
        "DDPFullyShardedPlugin",
        "HorovodPlugin",
        "IPUPlugin",
        "ParallelPlugin",
        "DDPShardedPlugin",
        "DDPSpawnShardedPlugin",
        "SingleDevicePlugin",
        "SingleTPUPlugin",
        "TPUSpawnPlugin",
        "TrainingTypePlugin",
    ),
)
@pytest.mark.parametrize("import_path", ("pytorch_lightning.plugins", "pytorch_lightning.plugins.training_type"))
def test_removed_training_type_plugin_classes(name, import_path):
    module = import_module(import_path)
    cls = getattr(module, name)
    with pytest.raises(NotImplementedError, match=f"{name}`.*no longer supported as of v1.8"):
        cls()


def test_removed_training_type_plugin_classes_inner_import():
    from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
    from pytorch_lightning.plugins.training_type.ddp2 import DDP2Plugin
    from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
    from pytorch_lightning.plugins.training_type.deepspeed import DeepSpeedPlugin
    from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin
    from pytorch_lightning.plugins.training_type.fully_sharded import DDPFullyShardedPlugin
    from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin
    from pytorch_lightning.plugins.training_type.ipu import IPUPlugin
    from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
    from pytorch_lightning.plugins.training_type.sharded import DDPShardedPlugin
    from pytorch_lightning.plugins.training_type.sharded_spawn import DDPSpawnShardedPlugin
    from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
    from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin
    from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin
    from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin

    with pytest.raises(NotImplementedError, match="DDPPlugin`.*no longer supported as of v1.8"):
        DDPPlugin()
    with pytest.raises(NotImplementedError, match="DDP2Plugin`.*no longer supported as of v1.8"):
        DDP2Plugin()
    with pytest.raises(NotImplementedError, match="DDPSpawnPlugin`.*no longer supported as of v1.8"):
        DDPSpawnPlugin()
    with pytest.raises(NotImplementedError, match="DeepSpeedPlugin`.*no longer supported as of v1.8"):
        DeepSpeedPlugin()
    with pytest.raises(NotImplementedError, match="DataParallelPlugin`.*no longer supported as of v1.8"):
        DataParallelPlugin()
    with pytest.raises(NotImplementedError, match="DDPFullyShardedPlugin`.*no longer supported as of v1.8"):
        DDPFullyShardedPlugin()
    with pytest.raises(NotImplementedError, match="HorovodPlugin`.*no longer supported as of v1.8"):
        HorovodPlugin()
    with pytest.raises(NotImplementedError, match="IPUPlugin`.*no longer supported as of v1.8"):
        IPUPlugin()
    with pytest.raises(NotImplementedError, match="ParallelPlugin`.*no longer supported as of v1.8"):
        ParallelPlugin()
    with pytest.raises(NotImplementedError, match="DDPShardedPlugin`.*no longer supported as of v1.8"):
        DDPShardedPlugin()
    with pytest.raises(NotImplementedError, match="DDPSpawnShardedPlugin`.*no longer supported as of v1.8"):
        DDPSpawnShardedPlugin()
    with pytest.raises(NotImplementedError, match="SingleDevicePlugin`.*no longer supported as of v1.8"):
        SingleDevicePlugin()
    with pytest.raises(NotImplementedError, match="SingleTPUPlugin`.*no longer supported as of v1.8"):
        SingleTPUPlugin()
    with pytest.raises(NotImplementedError, match="TPUSpawnPlugin`.*no longer supported as of v1.8"):
        TPUSpawnPlugin()
    with pytest.raises(NotImplementedError, match="TrainingTypePlugin`.*no longer supported as of v1.8"):
        TrainingTypePlugin()

    from pytorch_lightning.plugins.training_type.utils import on_colab_kaggle

    with pytest.raises(NotImplementedError, match="on_colab_kaggle`.*no longer supported as of v1.8"):
        on_colab_kaggle()

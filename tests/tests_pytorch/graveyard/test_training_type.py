from importlib import import_module

import pytest

from pytorch_lightning import Trainer


def test_removed_training_type_plugin_property():
    trainer = Trainer()
    with pytest.raises(RuntimeError, match="training_type_plugin` was removed"):
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
    with pytest.raises(RuntimeError, match=f"{name}` class was removed"):
        cls()


def test_removed_training_type_plugin_classes_inner_import():
    from pytorch_lightning.plugins.training_type.ddp import DDPPlugin

    with pytest.raises(RuntimeError, match="DDPPlugin` class was removed"):
        DDPPlugin()

    from pytorch_lightning.plugins.training_type.ddp2 import DDP2Plugin

    with pytest.raises(RuntimeError, match="DDP2Plugin` class was removed"):
        DDP2Plugin()

    from pytorch_lightning.plugins.training_type.ddp_spawn import DDP2Plugin

    with pytest.raises(RuntimeError, match="DDP2Plugin` class was removed"):
        DDP2Plugin()

    from pytorch_lightning.plugins.training_type.deepspeed import DeepSpeedPlugin

    with pytest.raises(RuntimeError, match="DeepSpeedPlugin` class was removed"):
        DeepSpeedPlugin()

    from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin

    with pytest.raises(RuntimeError, match="DataParallelPlugin` class was removed"):
        DataParallelPlugin()

    from pytorch_lightning.plugins.training_type.fully_sharded import DDPFullyShardedPlugin

    with pytest.raises(RuntimeError, match="DDPFullyShardedPlugin` class was removed"):
        DDPFullyShardedPlugin()

    from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin

    with pytest.raises(RuntimeError, match="HorovodPlugin` class was removed"):
        HorovodPlugin()

    from pytorch_lightning.plugins.training_type.ipu import IPUPlugin

    with pytest.raises(RuntimeError, match="IPUPlugin` class was removed"):
        IPUPlugin()

    from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin

    with pytest.raises(RuntimeError, match="ParallelPlugin` class was removed"):
        ParallelPlugin()

    from pytorch_lightning.plugins.training_type.sharded import DDPShardedPlugin

    with pytest.raises(RuntimeError, match="DDPShardedPlugin` class was removed"):
        DDPShardedPlugin()

    from pytorch_lightning.plugins.training_type.sharded_spawn import DDPSpawnShardedPlugin

    with pytest.raises(RuntimeError, match="DDPSpawnShardedPlugin` class was removed"):
        DDPSpawnShardedPlugin()

    from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin

    with pytest.raises(RuntimeError, match="SingleDevicePlugin` class was removed"):
        SingleDevicePlugin()

    from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin

    with pytest.raises(RuntimeError, match="SingleTPUPlugin` class was removed"):
        SingleTPUPlugin()

    from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin

    with pytest.raises(RuntimeError, match="TPUSpawnPlugin` class was removed"):
        TPUSpawnPlugin()

    from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin

    with pytest.raises(RuntimeError, match="TrainingTypePlugin` class was removed"):
        TrainingTypePlugin()

    from pytorch_lightning.plugins.training_type.utils import on_colab_kaggle

    with pytest.raises(RuntimeError, match="on_colab_kaggle` was removed"):
        on_colab_kaggle()

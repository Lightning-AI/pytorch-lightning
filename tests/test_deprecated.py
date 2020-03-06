"""Test deprecated functionality which will be removed in vX.Y.Z"""

from pytorch_lightning import Trainer


def test_to_be_removed_in_v0_8_0_module_imports():
    from pytorch_lightning.logging.comet_logger import CometLogger  # noqa: F811
    from pytorch_lightning.logging.mlflow_logger import MLFlowLogger  # noqa: F811
    from pytorch_lightning.logging.test_tube_logger import TestTubeLogger  # noqa: F811

    from pytorch_lightning.pt_overrides.override_data_parallel import (  # noqa: F811
        LightningDataParallel, LightningDistributedDataParallel)
    from pytorch_lightning.overrides.override_data_parallel import (  # noqa: F811
        LightningDataParallel, LightningDistributedDataParallel)

    from pytorch_lightning.core.model_saving import ModelIO  # noqa: F811
    from pytorch_lightning.core.root_module import LightningModule  # noqa: F811

    from pytorch_lightning.root_module.decorators import data_loader  # noqa: F811
    from pytorch_lightning.root_module.grads import GradInformation  # noqa: F811
    from pytorch_lightning.root_module.hooks import ModelHooks  # noqa: F811
    from pytorch_lightning.root_module.memory import ModelSummary  # noqa: F811
    from pytorch_lightning.root_module.model_saving import ModelIO  # noqa: F811
    from pytorch_lightning.root_module.root_module import LightningModule  # noqa: F811


def test_to_be_removed_in_v0_8_0_trainer():
    mapping_old_new = {
        'gradient_clip': 'gradient_clip_val',
        'nb_gpu_nodes': 'num_nodes',
        'max_nb_epochs': 'max_epochs',
        'min_nb_epochs': 'min_epochs',
        'nb_sanity_val_steps': 'num_sanity_val_steps',
    }
    # skip 0 since it may be interested as False
    kwargs = {k: (i + 1) for i, k in enumerate(mapping_old_new)}

    trainer = Trainer(**kwargs)

    for attr_old in mapping_old_new:
        attr_new = mapping_old_new[attr_old]
        assert kwargs[attr_old] == getattr(trainer, attr_old), \
            'Missing deprecated attribute "%s"' % attr_old
        assert kwargs[attr_old] == getattr(trainer, attr_new), \
            'Wrongly passed deprecated argument "%s" to attribute "%s"' % (attr_old, attr_new)


def test_to_be_removed_in_v0_9_0_module_imports():
    from pytorch_lightning.core.decorators import data_loader  # noqa: F811

    from pytorch_lightning.logging.comet import CometLogger  # noqa: F402
    from pytorch_lightning.logging.mlflow import MLFlowLogger  # noqa: F402
    from pytorch_lightning.logging.neptune import NeptuneLogger  # noqa: F402
    from pytorch_lightning.logging.test_tube import TestTubeLogger  # noqa: F402
    from pytorch_lightning.logging.wandb import WandbLogger  # noqa: F402

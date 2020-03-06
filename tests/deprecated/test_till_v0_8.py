"""Test deprecated functionality which will be removed in 0.8.0"""

from pytorch_lightning import Trainer

def test_module_imports():
    from pytorch_lightning.logging.comet import CometLogger
    from pytorch_lightning.logging.mlflow import MLFlowLogger
    from pytorch_lightning.logging.neptune import NeptuneLogger
    from pytorch_lightning.logging.test_tube import TestTubeLogger
    from pytorch_lightning.logging.wandb import WandbLogger

    from pytorch_lightning.logging.comet_logger import CometLogger
    from pytorch_lightning.logging.mlflow_logger import MLFlowLogger
    from pytorch_lightning.logging.test_tube_logger import TestTubeLogger

    from pytorch_lightning.pt_overrides.override_data_parallel import (
        LightningDataParallel, LightningDistributedDataParallel)
    from pytorch_lightning.overrides.override_data_parallel import (
        LightningDataParallel, LightningDistributedDataParallel)

    from pytorch_lightning.core.model_saving import ModelIO
    from pytorch_lightning.core.root_module import LightningModule

    # from pytorch_lightning.root_module. import LightningModule


def test_trainer_args():
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

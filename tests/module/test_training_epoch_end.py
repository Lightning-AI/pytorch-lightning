import torch

from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate

import tests.base.utils as tutils


def test_training_epoch_end_metrics_collection(tmpdir):

    class CurrentModel(EvalModelTemplate):

        def training_epoch_end(self, outputs):
            return {
                'progress_bar': {
                    'new_metric': torch.tensor(3)
                }
            }

    model = CurrentModel(tutils.get_default_hparams())
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        train_percent_check=0.001,
        val_percent_check=0.01,
    )
    trainer.fit(model)


    assert trainer.progress_bar_dict['new_metric'] == 3

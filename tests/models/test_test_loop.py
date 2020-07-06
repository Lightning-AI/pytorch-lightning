import os
import pytorch_lightning as pl
from tests.base import EvalModelTemplate
import tests.base.develop_utils as tutils


def test_ddp_spawn_test(tmpdir):
    tutils.set_random_master_port()

    model = EvalModelTemplate()
    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=2,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='ddp_spawn',
    )
    trainer.fit(model)
    assert 'ckpt' in trainer.checkpoint_callback.best_model_path
    results = trainer.test()
    assert 'test_acc' in results

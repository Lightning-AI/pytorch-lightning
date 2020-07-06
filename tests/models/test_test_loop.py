import os
import pytorch_lightning as pl
from tests.base import EvalModelTemplate
from tests.base.develop_utils import pl_multi_process_test


@pl_multi_process_test
def test_ddp(tmpdir):

    model = EvalModelTemplate()
    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=2,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='ddp',
    )
    trainer.fit(model)
    assert 'ckpt' in trainer.checkpoint_callback.best_model_path
    results = trainer.test()
    assert 'test_acc' in results


def test_ddp_spawn_test(tmpdir):

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

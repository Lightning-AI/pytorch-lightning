import pytest

from pytorch_lightning.trainer import Trainer
from tests.base import SimpleModule


@pytest.mark.parametrize('steps', [1, 5])
def test_trainer_max_steps(tmpdir, steps):

    model = SimpleModule()
    trainer = Trainer(
        default_root_dir=tmpdir,
        logger=False,
        limit_train_batches=1,
        limit_val_batches=1,
        max_steps=steps,
    )
    trainer.fit(model)

    assert trainer.global_step == steps

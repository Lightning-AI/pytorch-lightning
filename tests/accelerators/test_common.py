import pytest
import torch

from pytorch_lightning import Trainer
from tests.accelerators.test_dp import CustomClassificationModelDP
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf


@pytest.mark.parametrize("trainer_kwargs", (
    pytest.param({"gpus": 1}, marks=RunIf(min_gpus=1)),
    pytest.param({"accelerator": "dp", "gpus": 2}, marks=RunIf(min_gpus=2)),
    pytest.param({"accelerator": "ddp_spawn", "gpus": 2}, marks=RunIf(min_gpus=2)),
))
def test_evaluate(tmpdir, trainer_kwargs, tutils=None):
    tutils.set_random_master_port()

    dm = ClassifDataModule()
    model = CustomClassificationModelDP()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=10,
        limit_val_batches=10,
        deterministic=True,
        **trainer_kwargs
    )

    result = trainer.fit(model, datamodule=dm)
    assert result
    assert 'ckpt' in trainer.checkpoint_callback.best_model_path

    old_weights = model.layer_0.weight.clone().detach().cpu()

    result = trainer.validate(datamodule=dm)
    assert result[0]['val_acc'] > 0.7

    result = trainer.test(datamodule=dm)
    assert result[0]['test_acc'] > 0.6

    # make sure weights didn't change
    new_weights = model.layer_0.weight.clone().detach().cpu()
    assert torch.testing.assert_allclose(old_weights, new_weights)

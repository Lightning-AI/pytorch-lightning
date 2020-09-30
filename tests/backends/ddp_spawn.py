import pytest
import torch

import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from tests.base import EvalModelTemplate
from pytorch_lightning.core import memory


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_early_stop_ddp_spawn(tmpdir):
    """Make sure DDP works. with early stopping"""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        early_stop_callback=True,
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='ddp_spawn',
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_model_ddp_spawn(tmpdir):
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='ddp_spawn',
        progress_bar_refresh_rate=0
    )

    model = EvalModelTemplate()

    tpipes.run_model_test(trainer_options, model)

    # test memory helper functions
    memory.get_memory_profile('min_max')


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_ddp_all_dataloaders_passed_to_fit(tmpdir):
    """Make sure DDP works with dataloaders passed to fit()"""
    tutils.set_random_master_port()

    model = EvalModelTemplate()
    fit_options = dict(train_dataloader=model.train_dataloader(),
                       val_dataloaders=model.val_dataloader())

    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        gpus=[0, 1],
        distributed_backend='ddp_spawn'
    )
    result = trainer.fit(model, **fit_options)
    assert result == 1, "DDP doesn't work with dataloaders passed to fit()."

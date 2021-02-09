import os
import sys

import numpy as np
import pytest
import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities import AllGatherGrad
from tests.helpers.boring_model import BoringModel


def setup_ddp(rank, world_size):
    """ Setup ddp enviroment """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8088"

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _test_all_gather_ddp(rank, world_size):
    setup_ddp(rank, world_size)

    tensor1 = torch.ones(8, requires_grad=True)
    tensor2 = torch.ones((8, 16, 32), requires_grad=True)

    tensor1_gathered = AllGatherGrad.apply(tensor1)
    tensor2_gathered = AllGatherGrad.apply(tensor2)

    tensor1_gathered = tensor1_gathered * rank
    tensor2_gathered = tensor2_gathered * rank

    tensor1_gathered.sum().backward()
    tensor2_gathered.sum().backward()

    grad1 = torch.zeros_like(tensor1.grad).fill_(torch.arange(world_size).sum().float())
    grad2 = torch.zeros_like(tensor2.grad).fill_(torch.arange(world_size).sum().float())

    assert torch.allclose(grad1, tensor1.grad)
    assert torch.allclose(grad2, tensor2.grad)


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_all_gather_ddp():
    world_size = 3
    torch.multiprocessing.spawn(_test_all_gather_ddp, args=(world_size, ), nprocs=world_size)


@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(
    not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest"
)
def test_all_gather_collection(tmpdir):

    class TestModel(BoringModel):

        training_epoch_end_called = False

        def training_epoch_end(self, outputs) -> None:
            self.training_epoch_end_called = True
            losses = torch.stack([x["loss"] for x in outputs])
            gathered_loss = self.all_gather({
                "losses_np_ndarray": np.array([1, 2, 3]),
                "losses_bool": [True, False],
                "losses_float": [0., 1., 2.],
                "losses_int": [0, 1, 2],
                "losses": losses,
                "losses_list": [losses, losses]
            })
            assert gathered_loss["losses_np_ndarray"][0].dtype == torch.int64
            # torch.bool can't be all_gathered
            assert gathered_loss["losses_bool"][0].dtype == torch.uint8
            assert gathered_loss["losses_float"][0].dtype == torch.float
            assert gathered_loss["losses_int"][0].dtype == torch.int
            assert gathered_loss["losses_list"][0].numel() == 2 * len(losses)
            assert gathered_loss["losses"].numel() == 2 * len(losses)

    seed_everything(42)

    model = TestModel()

    limit_train_batches = 8
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        enable_pl_optimizer=True,
        gpus=2,
        accelerator="ddp",
    )

    trainer.fit(model)
    assert model.training_epoch_end_called

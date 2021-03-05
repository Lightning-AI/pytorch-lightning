from types import ModuleType
from typing import Any, List, Optional, Union

import torch
from torch import nn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.imports import _PYSYFT_AVAILABLE
from tests.helpers.runif import RunIf

if _PYSYFT_AVAILABLE:
    import syft as sy
    from syft.ast.module import Module


@RunIf(syft=True)
def test_syft(tmpdir):
    duet = sy.VirtualMachine(name="alice").get_root_client()
    SyModuleProxyType = Union[ModuleType, Module]
    SyModelProxyType = Union[torch.nn.Module, sy.Module]

    # cant use lib_ast during test search time
    TorchTensorPointerType = Any  # sy.lib_ast.torch.Tensor.pointer_type
    SyTensorProxyType = Union[torch.Tensor, TorchTensorPointerType]  # type: ignore

    sy.logger.remove()
    alice = sy.VirtualMachine(name="alice")
    duet = alice.get_root_client()
    # bookkeeping
    sy.client_cache["duet"] = duet

    class BoringSyNet(sy.Module):

        def __init__(self, torch_ref: SyModuleProxyType) -> None:
            super(BoringSyNet, self).__init__(torch_ref=torch_ref)
            self.fc2 = self.torch_ref.nn.Linear(32, 2)

        def forward(self, x: SyTensorProxyType) -> SyTensorProxyType:
            return self.fc2(x)

    class BoringModel(LightningModule):

        def __init__(
            self,
            local_torch: ModuleType,
            download_back: bool = False,
            run_locally: bool = False,
        ) -> None:
            super().__init__()
            self.local_model = BoringSyNet(local_torch)
            # Those are helpers to easily work with `sy.Module`
            self.remote_torch = duet.torch
            self.local_torch = local_torch
            self.download_back = download_back
            self.run_locally = run_locally
            self.get = self.local_model.get
            self.send = self.local_model.send

        def is_remote(self) -> bool:
            # Training / Evaluation is done remotely and Testing is done locally unless run_locally is True
            if self.run_locally or (not self.trainer.training and self.trainer.evaluation_loop.testing):
                return False
            return True

        @property
        def torch(self) -> SyModuleProxyType:
            return self.remote_torch if self.is_remote() else self.local_torch

        @property
        def model(self) -> SyModelProxyType:
            if self.is_remote():
                return self.remote_model
            else:
                if self.download_back:
                    return self.get_model()
                else:
                    return self.local_model

        def send_model(self) -> None:
            self.remote_model = self.local_model.send(duet)

        def get_model(self) -> type(nn.Module):  # type: ignore
            return self.remote_model.get(request_block=True)

        def forward(self, x: SyTensorProxyType) -> SyTensorProxyType:
            return self.model(x)

        def training_step(self, batch: SyTensorProxyType, batch_idx: Optional[int]) -> SyTensorProxyType:
            data_ptr = batch
            output = self.forward(data_ptr)
            return self.torch.nn.functional.mse_loss(output, self.torch.ones_like(output))

        def test_step(self, batch: SyTensorProxyType, batch_idx: Optional[int]) -> None:
            output = self.forward(batch)
            loss = self.loss(output, self.torch.ones_like(output))
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        def configure_optimizers(self) -> List:
            optimizer = self.torch.optim.SGD(self.model.parameters(), lr=0.1)  # type: ignore
            return [optimizer]

        def train_dataloader(self):
            return self.torch.utils.data.DataLoader(self.torch.randn(64, 32))

    model = BoringModel(torch, download_back=False)
    model.send_model()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=4,
        limit_test_batches=4,
    )

    trainer.fit(model)

    model.download_back = True
    trainer.test(model)

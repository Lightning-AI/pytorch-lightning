# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict

import torch
from torch.nn import Module
from torch.optim import Optimizer

from pytorch_lightning.plugins.secure.base import BaseSecurePlugin
from pytorch_lightning.utilities.imports import _PYSYFT_AVAILABLE


class PySyftPlugin(BaseSecurePlugin):

    @staticmethod
    def optimizer_state(optimizer: Optimizer) -> Dict[str, torch.Tensor]:
        """
        Returns state of an optimizer. Allows for syncing/collating optimizer state from processes in custom
        plugins.
        """
        return {}

    @staticmethod
    def save_function(trainer, filepath: str, save_weights_only: bool) -> None:
        model_ref = trainer.lightning_module
        sy_model = model_ref.get_model()
        sy_model.device = model_ref.device
        sy_model.hparams = model_ref.hparams
        sy_model.on_save_checkpoint = model_ref.on_save_checkpoint
        trainer.training_type_plugin.model = sy_model
        # todo (tudorcebere): Add support for optimizer and scheduler states   # noqa E265
        trainer.accelerator.optimizer_state = PySyftPlugin.optimizer_state
        trainer.save_checkpoint(filepath, save_weights_only)
        trainer.training_type_plugin.model = model_ref


if _PYSYFT_AVAILABLE:

    from types import ModuleType
    from typing import Any, Union

    import syft as sy

    from pytorch_lightning.core.lightning import LightningModule

    SyModuleProxyType = Union[ModuleType, Module]
    SyModelProxyType = Union[Module, sy.Module]

    # cant use lib_ast during test search time
    TorchTensorPointerType = Any  # sy.lib_ast.torch.Tensor.pointer_type
    SyTensorProxyType = Union[torch.Tensor, TorchTensorPointerType]  # type: ignore

    class SyLightningModule(LightningModule):

        def __init__(self, download_back: bool = False, run_locally: bool = False) -> None:
            super().__init__()
            # Those are helpers to easily work with `sy.Module`
            self.duet = sy.client_cache["duet"]
            self.remote_torch = sy.client_cache["duet"].torch
            self.local_torch = globals()["torch"]
            self.download_back = download_back
            self.run_locally = run_locally

        def setup(self, stage_name: str):
            self.get = self.module.get
            self.send = self.module.send
            self.send_model()

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
                    return self.module

        def send_model(self) -> None:
            self.remote_model = self.module.send(self.duet)

        def get_model(self) -> type(Module):  # type: ignore
            return self.remote_model.get(request_block=True)

        def parameters(self):
            return self.module.parameters()

        def state_dict(self):
            return self.module.state_dict()

        def load_state_dict(self, state_dict, strict: bool = True):
            return self.module.load_state_dict(state_dict)

        def forward(self, x: SyTensorProxyType) -> SyTensorProxyType:
            return self.model(x)

        def on_train_start(self) -> None:
            self.download_back = False

        def on_test_start(self) -> None:
            self.download_back = True

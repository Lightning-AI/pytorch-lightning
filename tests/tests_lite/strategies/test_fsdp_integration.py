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
import tempfile

import pytest
import torch
from tests_lite.helpers.models import RandomDataset
from tests_lite.helpers.runif import RunIf
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import wrap
from torch.utils.data import DataLoader

from lightning_lite import LightningLite
from lightning_lite.plugins import FSDPPrecision
from lightning_lite.strategies import FSDPStrategy


class FSDPLite(LightningLite):
    manual_wrapping = False

    def run(self):
        model = self.get_model()

        dataloader = DataLoader(RandomDataset(32, 64))

        # model needs to be set up first in FSDP
        model = self.setup_model(model)

        # get parameters on the wrapped model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # optimizer nees to be set up independently
        optimizer = self.setup_optimizers(optimizer)

        dataloader = self.setup_dataloaders(dataloader)
        model.train()

        data_iter = iter(dataloader)
        batch = next(data_iter)
        loss = self.step(model, batch)
        self.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        with tempfile.TemporaryFile() as ckpt_path:
            ckpt_path = self.broadcast(str(ckpt_path))
            self._strategy.save_checkpoint(model.state_dict(), ckpt_path)

        self._assert_save_equality(model, ckpt_path)

    def get_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        if not self.manual_wrapping:
            return model

        for i, layer in enumerate(model):
            if i % 2 == 0:
                model[i] = wrap(layer)
        model = wrap(model)
        return model

    def step(self, model, batch):
        forward_module = model._forward_module
        original_module = model.module
        assert isinstance(forward_module, FullyShardedDataParallel)
        assert isinstance(self._precision, FSDPPrecision)

        precision = torch.float16 if self._precision.precision == 16 else torch.bfloat16
        assert forward_module.mixed_precision.param_dtype == precision
        assert forward_module.mixed_precision.reduce_dtype == precision
        assert forward_module.mixed_precision.buffer_dtype == precision

        for layer_num in [0, 2]:
            assert isinstance(original_module[layer_num], FullyShardedDataParallel)
            assert original_module[layer_num].mixed_precision.param_dtype == precision
            assert original_module[layer_num].mixed_precision.reduce_dtype == precision
            assert original_module[layer_num].mixed_precision.buffer_dtype == precision

        output = model(batch)
        loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
        return loss

    def _assert_save_equality(self, model, ckpt_path):
        current_state_dict = self._strategy.get_module_state_dict(model)

        checkpoint = self.load(ckpt_path)
        loaded_model = self.get_model()
        loaded_model.load_state_dict(checkpoint)

        # model parameters are identical after loading
        for current_param, loaded_param in zip(current_state_dict.values(), loaded_model.state_dict().values()):
            assert torch.equal(current_param.float().cpu(), loaded_param.cpu())


def custom_auto_wrap_policy(module, recurse, unwrapped_params: int, min_num_params: int = int(1e8)) -> bool:
    return unwrapped_params >= 2


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("precision", (16, pytest.param("bf16", marks=RunIf(bf16_cuda=True))))
@pytest.mark.parametrize("manual_wrapping", [True, False])
def test_fsdp_train_save_load(manual_wrapping, precision):
    """Test FSDP training, saving and loading with different wrapping and precision settings."""
    strategy = FSDPStrategy() if manual_wrapping else FSDPStrategy(auto_wrap_policy=custom_auto_wrap_policy)
    lite = FSDPLite(accelerator="cuda", strategy=strategy, devices=2, precision=precision)
    lite.manual_wrapping = manual_wrapping
    lite.run()

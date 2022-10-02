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
from tests_lite.helpers.models import BoringLite
from tests_lite.helpers.runif import RunIf
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import wrap

from lightning_lite.plugins import FSDPPrecision


class FSDPLite(BoringLite):
    manual_wrapping = False

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
        assert isinstance(self._precision_plugin, FSDPPrecision)

        precision = torch.float16 if self._precision_plugin.precision == 16 else torch.bfloat16
        assert forward_module.mixed_precision.param_dtype == precision
        assert forward_module.mixed_precision.reduce_dtype == precision
        assert forward_module.mixed_precision.buffer_dtype == precision

        for layer_num in [0, 2]:
            if self.manual_wrapping:
                assert isinstance(original_module[layer_num], FullyShardedDataParallel)
            else:
                assert isinstance(forward_module[layer_num], FullyShardedDataParallel)

            assert original_module[layer_num].mixed_precision.param_dtype == precision
            assert original_module[layer_num].mixed_precision.reduce_dtype == precision
            assert original_module[layer_num].mixed_precision.buffer_dtype == precision

        return super().step(model, batch)

    def run(self):
        super().run()
        with tempfile.TemporaryFile() as ckpt_path:
            ckpt_path = self.broadcast(str(ckpt_path))
            self._strategy.save_checkpoint(self.model.state_dict(), ckpt_path)

        self._assert_save_equality(ckpt_path)

    def _assert_save_equality(self, ckpt_path):
        current_state_dict = self._strategy.get_module_state_dict(self.model)

        checkpoint = self.load(ckpt_path)
        loaded_model = self.get_model()
        loaded_model.load_state_dict(checkpoint)

        # model parameters are identical after loading
        for current_param, loaded_param in zip(current_state_dict.values(), loaded_model.state_dict().values()):
            assert torch.equal(current_param.float().cpu(), loaded_param.cpu())


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("precision", (16, pytest.param("bf16", marks=RunIf(bf16_cuda=True))))
@pytest.mark.parametrize("manual_wrapping", [True, False])
def test_fsdp_train_save_load(manual_wrapping, precision):
    """Test to ensure that checkpoint is saved correctly when using a single GPU, and all stages can be run."""
    lite = FSDPLite(accelerator="cuda", strategy="fsdp", devices=2, precision=precision)
    lite.manual_wrapping = manual_wrapping
    lite.run()


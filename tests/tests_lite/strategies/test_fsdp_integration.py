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
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import wrap

from lightning_lite.plugins import FSDPPrecision
from tests.tests_lite.helpers.runif import RunIf


class FSDPLite(BoringLite):
    def get_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        for i, layer in enumerate(model):
            if i % 2 == 0:
                model[i] = wrap(layer)

    def step(self, model, batch):
        forward_module = model._forward_module
        original_module = model.module
        assert isinstance(forward_module, FullyShardedDataParallel)
        assert isinstance(self._precision_plugin, FSDPPrecision)
        # the root module should not be resharding
        assert forward_module.reshard_after_forward is False

        precision = torch.float16 if self._precision_plugin.precision == 16 else torch.bfloat16
        assert forward_module.mixed_precision.param_dtype == precision
        assert forward_module.mixed_precision.reduce_dtype == precision
        assert forward_module.mixed_precision.buffer_dtype == precision

        for layer_num in [0, 2]:
            assert isinstance(original_module[layer_num], FullyShardedDataParallel)
            # The nested layers should have `reshard_after_forward` set to True
            assert original_module[layer_num].reshard_after_forward

            assert original_module[layer_num].mixed_precision.param_dtype == precision
            assert original_module[layer_num].mixed_precision.reduce_dtype == precision
            assert original_module[layer_num].mixed_precision.buffer_dtype == precision

        return super().step(model, batch)

    def run(self):
        super().run()
        with tempfile.TemporaryFile() as ckpt_path:
            ckpt_path = self.broadcast(str(ckpt_path))


            checkpoint = dict(
                model=self.model.state_dict(),
                optimizer = self.optimizer.state_dict()
            )

            self._strategy.save_checkpoint(checkpoint, ckpt_path)

        _assert_save_equality(self, ckpt_path)

#
#
# class TestFSDPModelAutoWrapped(BoringModel):
#     def __init__(self):
#         super().__init__()
#         self.layer = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
#
#     def configure_optimizers(self):
#         return torch.optim.SGD(self.trainer.model.parameters(), lr=0.1)
#
#     def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
#         self._assert_layer_fsdp_instance()
#
#     def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
#         self._assert_layer_fsdp_instance()
#
#     def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
#         self._assert_layer_fsdp_instance()
#
#     def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
#         self._assert_layer_fsdp_instance()
#
#     def _assert_layer_fsdp_instance(self) -> None:
#         assert isinstance(self.layer, torch.nn.Sequential)
#         assert isinstance(self.trainer.strategy.precision_plugin, FullyShardedNativeNativeMixedPrecisionPlugin)
#
#         precision = torch.float16 if self.precision == 16 else torch.bfloat16
#         for layer_num in [0, 2]:
#             assert isinstance(self.layer[layer_num], FullyShardedDataParallel)
#             # Assert that the nested layers are set reshard_after_forward to True
#             assert self.layer[layer_num].reshard_after_forward
#
#             assert self.layer[layer_num].mixed_precision.param_dtype == precision
#             assert self.layer[layer_num].mixed_precision.reduce_dtype == precision
#             assert self.layer[layer_num].mixed_precision.buffer_dtype == precision



@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("precision", (16, pytest.param("bf16", marks=RunIf(bf16_cuda=True))))
def test_fsdp_train_save_load(precision):
    """Test to ensure that checkpoint is saved correctly when using a single GPU, and all stages can be run."""
    FSDPLite(accelerator="cuda", strategy="fsdp", devices=2, precision=precision).run()


def _assert_save_equality(lite, ckpt_path):
    model_state_dict = lite._strategy.get_module_state_dict()

    if lite.is_global_zero:
        checkpoint = lite.load(ckpt_path)
        saved_model = lite.get_model().load_state_dict(checkpoint["state_dict"])

        # model parameters are identical after loading
        for ddp_param, shard_param in zip(model_state_dict.values(), saved_model.state_dict().values()):
            assert torch.equal(ddp_param.float().cpu(), shard_param)

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
"""Test deprecated functionality which will be removed in vX.Y.Z"""
import sys

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.overrides.data_parallel import (
    LightningDataParallel,
    LightningDistributedDataParallel,
    LightningParallelModule,
)
from pytorch_lightning.overrides.distributed import LightningDistributedModule
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from tests.deprecated_api import _soft_unimport_module
from tests.helpers import BoringModel


def test_v1_4_0_deprecated_trainer_attributes():
    with pytest.deprecated_call(match="will be removed in v1.4."):
        trainer = Trainer()
        _ = trainer.accelerator_backend
    assert trainer.accelerator == trainer.accelerator_backend


def test_v1_4_0_deprecated_trainer_methods():
    with pytest.deprecated_call(match='will be removed in v1.4'):
        trainer = Trainer()
        _ = trainer.get_model()
    assert trainer.get_model() == trainer.lightning_module


def test_v1_4_0_deprecated_imports():
    _soft_unimport_module('pytorch_lightning.utilities.argparse_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.argparse_utils import from_argparse_args  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.model_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.model_utils import is_overridden  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.warning_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.warning_utils import WarningCache  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.xla_device_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils  # noqa: F811 F401


def test_v1_4_0_deprecated_trainer_device_distrib():
    """Test that Trainer attributes works fine."""
    trainer = Trainer()
    trainer.accelerator_connector._distrib_type = None
    trainer.accelerator_connector._device_type = None

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.on_cpu = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.on_cpu

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.on_gpu = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.on_gpu

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.on_tpu = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.on_tpu
    trainer.accelerator_connector._device_type = None
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_tpu = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_tpu

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_dp = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_dp

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_ddp = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_ddp

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_ddp2 = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_ddp2

    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        trainer.use_horovod = True
    with pytest.deprecated_call(match='deprecated in v1.2 and will be removed in v1.4'):
        assert trainer.use_horovod


def test_v1_4_0_deprecated_metrics():
    from pytorch_lightning.metrics.functional.classification import stat_scores_multiple_classes
    with pytest.deprecated_call(match='will be removed in v1.4'):
        stat_scores_multiple_classes(pred=torch.tensor([0, 1]), target=torch.tensor([0, 1]))

    from pytorch_lightning.metrics.functional.classification import iou
    with pytest.deprecated_call(match='will be removed in v1.4'):
        iou(torch.randint(0, 2, (10, 3, 3)), torch.randint(0, 2, (10, 3, 3)))

    from pytorch_lightning.metrics.functional.classification import recall
    with pytest.deprecated_call(match='will be removed in v1.4'):
        recall(torch.randint(0, 2, (10, 3, 3)), torch.randint(0, 2, (10, 3, 3)))

    from pytorch_lightning.metrics.functional.classification import precision
    with pytest.deprecated_call(match='will be removed in v1.4'):
        precision(torch.randint(0, 2, (10, 3, 3)), torch.randint(0, 2, (10, 3, 3)))

    from pytorch_lightning.metrics.functional.classification import precision_recall
    with pytest.deprecated_call(match='will be removed in v1.4'):
        precision_recall(torch.randint(0, 2, (10, 3, 3)), torch.randint(0, 2, (10, 3, 3)))

    # Testing deprecation of class_reduction arg in the *new* precision
    from pytorch_lightning.metrics.functional import precision
    with pytest.deprecated_call(match='will be removed in v1.4'):
        precision(torch.randint(0, 2, (10, )), torch.randint(0, 2, (10, )), class_reduction='micro')

    # Testing deprecation of class_reduction arg in the *new* recall
    from pytorch_lightning.metrics.functional import recall
    with pytest.deprecated_call(match='will be removed in v1.4'):
        recall(torch.randint(0, 2, (10, )), torch.randint(0, 2, (10, )), class_reduction='micro')

    from pytorch_lightning.metrics.functional.classification import auc
    with pytest.deprecated_call(match='will be removed in v1.4'):
        auc(torch.rand(10, ).sort().values, torch.rand(10, ))

    from pytorch_lightning.metrics.functional.classification import auroc
    with pytest.deprecated_call(match='will be removed in v1.4'):
        auroc(torch.rand(10, ), torch.randint(0, 2, (10, )))

    from pytorch_lightning.metrics.functional.classification import multiclass_auroc
    with pytest.deprecated_call(match='will be removed in v1.4'):
        multiclass_auroc(torch.rand(20, 5).softmax(dim=-1), torch.randint(0, 5, (20, )), num_classes=5)

    from pytorch_lightning.metrics.functional.classification import auc_decorator
    with pytest.deprecated_call(match='will be removed in v1.4'):
        auc_decorator()

    from pytorch_lightning.metrics.functional.classification import multiclass_auc_decorator
    with pytest.deprecated_call(match='will be removed in v1.4'):
        multiclass_auc_decorator()


class CustomDDPPlugin(DDPSpawnPlugin):

    def configure_ddp(self):
        # old, deprecated implementation
        with pytest.deprecated_call(
            match='`LightningDistributedDataParallel` is deprecated since v1.2 and will be removed in v1.4.'
        ):
            self._model = LightningDistributedDataParallel(
                module=self.lightning_module,
                device_ids=self.determine_ddp_device_ids(),
                **self._ddp_kwargs,
            )
            assert isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            assert isinstance(self.model.module, LightningDistributedModule)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
def test_v1_4_0_deprecated_lightning_distributed_data_parallel(tmpdir):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        gpus=2,
        accelerator="ddp_spawn",
        plugins=[
            CustomDDPPlugin(
                parallel_devices=[torch.device("cuda", 0), torch.device("cuda", 1)],
                cluster_environment=TorchElasticEnvironment(),
            )
        ]
    )
    trainer.fit(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_v1_4_0_deprecated_lightning_data_parallel():
    model = BoringModel()
    with pytest.deprecated_call(match="`LightningDataParallel` is deprecated since v1.2 and will be removed in v1.4."):
        dp_model = LightningDataParallel(model, device_ids=[0])
    assert isinstance(dp_model, torch.nn.DataParallel)
    assert isinstance(dp_model.module, LightningParallelModule)


def test_v1_4_0_deprecated_manual_optimization_optimizer(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, batch, *_, **kwargs):
            opt = self.optimizers()
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.manual_backward(loss, opt)

        @property
        def automatic_optimization(self):
            return False

    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )
    with pytest.deprecated_call(
        match="`optimizer` argument to `manual_backward` is deprecated in v1.2 and will be removed in v1.4"
    ):
        trainer.fit(model)


def test_v1_4_0_deprecated_checkpoint_on(tmpdir):
    from pytorch_lightning.callbacks.model_checkpoint import warning_cache
    warning_cache.clear()

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            self.log("val_loss", -batch_idx)
            return super().training_step(batch, batch_idx)

    trainer = Trainer(default_root_dir=tmpdir, checkpoint_callback=True, max_epochs=1)

    with pytest.warns(DeprecationWarning, match=r"Relying on.*is deprecated in v1.2 and will be removed in v1.4"):
        trainer.fit(TestModel())

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
""" Test deprecated functionality which will be removed in v1.6.0 """
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.plugins.training_type import DDPPlugin, DDPSpawnPlugin
from tests.helpers import BoringDataModule, BoringModel


def test_v1_6_0_trainer_model_hook_mixin(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, checkpoint_callback=False, logger=False)
    trainer.fit(model)
    with pytest.deprecated_call(match="is deprecated in v1.4 and will be removed in v1.6"):
        trainer.is_function_implemented("training_step", model)

    with pytest.deprecated_call(match="is deprecated in v1.4 and will be removed in v1.6"):
        trainer.has_arg("training_step", "batch")


def test_old_transfer_batch_to_device_hook(tmpdir):

    class OldModel(BoringModel):

        def transfer_batch_to_device(self, batch, device):
            return super().transfer_batch_to_device(batch, device, None)

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=0, max_epochs=1)
    with pytest.deprecated_call(match='old signature will be removed in v1.6'):
        trainer.fit(OldModel())


def test_v1_6_0_ddp_num_nodes():
    with pytest.deprecated_call(match="Argument `num_nodes` in `DDPPlugin` is deprecated in v1.4"):
        DDPPlugin(num_nodes=1)


def test_v1_6_0_ddp_sync_batchnorm():
    with pytest.deprecated_call(match="Argument `sync_batchnorm` in `DDPPlugin` is deprecated in v1.4"):
        DDPPlugin(sync_batchnorm=False)


def test_v1_6_0_ddp_spawn_num_nodes():
    with pytest.deprecated_call(match="Argument `num_nodes` in `DDPPlugin` is deprecated in v1.4"):
        DDPSpawnPlugin(num_nodes=1)


def test_v1_6_0_ddp_spawn_sync_batchnorm():
    with pytest.deprecated_call(match="Argument `sync_batchnorm` in `DDPPlugin` is deprecated in v1.4"):
        DDPSpawnPlugin(sync_batchnorm=False)


def test_v1_6_0_tbptt_reduce_fx(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, *args):
            self.log("foo", 1, tbptt_reduce_fx=lambda x: x)
            return super().training_step(*args)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.deprecated_call(match=r"tbptt_reduce_fx=...\)` is no longer supported"):
        trainer.fit(TestModel())


def test_v1_6_0_tbptt_pad_token(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, *args):
            self.log("foo", 1, tbptt_pad_token=0)
            return super().training_step(*args)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    with pytest.deprecated_call(match=r"tbptt_pad_token=...\)` is no longer supported"):
        trainer.fit(TestModel())


def test_v1_6_0_datamodule_lifecycle_properties(tmpdir):
    dm = BoringDataModule()
    with pytest.deprecated_call(match=r"DataModule property `has_prepared_data` was deprecated in v1.4"):
        dm.has_prepared_data
    with pytest.deprecated_call(match=r"DataModule property `has_setup_fit` was deprecated in v1.4"):
        dm.has_setup_fit
    with pytest.deprecated_call(match=r"DataModule property `has_setup_validate` was deprecated in v1.4"):
        dm.has_setup_validate
    with pytest.deprecated_call(match=r"DataModule property `has_setup_test` was deprecated in v1.4"):
        dm.has_setup_test
    with pytest.deprecated_call(match=r"DataModule property `has_setup_predict` was deprecated in v1.4"):
        dm.has_setup_predict
    with pytest.deprecated_call(match=r"DataModule property `has_teardown_fit` was deprecated in v1.4"):
        dm.has_teardown_fit
    with pytest.deprecated_call(match=r"DataModule property `has_teardown_validate` was deprecated in v1.4"):
        dm.has_teardown_validate
    with pytest.deprecated_call(match=r"DataModule property `has_teardown_test` was deprecated in v1.4"):
        dm.has_teardown_test
    with pytest.deprecated_call(match=r"DataModule property `has_teardown_predict` was deprecated in v1.4"):
        dm.has_teardown_predict


def test_v1_6_0_datamodule_hooks_calls(tmpdir):
    """Test that repeated calls to DataHooks' hooks show a warning about the coming API change."""

    class TestDataModule(BoringDataModule):
        setup_calls = []
        teardown_calls = []
        prepare_data_calls = 0

        def setup(self, stage=None):
            super().setup(stage=stage)
            self.setup_calls.append(stage)

        def teardown(self, stage=None):
            super().teardown(stage=stage)
            self.teardown_calls.append(stage)

        def prepare_data(self):
            super().prepare_data()
            self.prepare_data_calls += 1

    dm = TestDataModule()
    dm.prepare_data()
    dm.prepare_data()
    dm.setup('fit')
    with pytest.deprecated_call(
        match=r"DataModule.setup has already been called, so it will not be called again. "
        "In v1.6 this behavior will change to always call DataModule.setup"
    ):
        dm.setup('fit')
    dm.setup()
    dm.setup()
    dm.teardown('validate')
    with pytest.deprecated_call(
        match=r"DataModule.teardown has already been called, so it will not be called again. "
        "In v1.6 this behavior will change to always call DataModule.teardown"
    ):
        dm.teardown('validate')

    assert dm.prepare_data_calls == 1
    assert dm.setup_calls == ['fit', None]
    assert dm.teardown_calls == ['validate']

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    trainer.test(BoringModel(), datamodule=dm)

    # same number of calls
    assert dm.prepare_data_calls == 1
    assert dm.setup_calls == ['fit', None]
    assert dm.teardown_calls == ['validate', 'test']

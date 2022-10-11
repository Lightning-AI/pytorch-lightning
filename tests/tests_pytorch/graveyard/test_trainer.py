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

import pytest

from pytorch_lightning import Trainer


@pytest.mark.parametrize(
    "attribute",
    [
        "gpus",
        "num_gpus",
        "root_gpu",
        "devices",
        "tpu_cores",
        "ipus",
        "use_amp",
        "weights_save_path",
        "lightning_optimizers",
        "should_rank_save_checkpoint",
        "validated_ckpt_path",
        "tested_ckpt_path",
        "predicted_ckpt_path",
        "verbose_evaluate",
    ],
)
def test_v2_0_0_unsupported_getters(attribute):
    trainer = Trainer()
    with pytest.raises(
        AttributeError, match=f"`Trainer.{attribute}` was deprecated in v1.6 and is no longer accessible as of v1.8."
    ):
        getattr(trainer, attribute)


@pytest.mark.parametrize(
    "attribute",
    [
        "validated_ckpt_path",
        "tested_ckpt_path",
        "predicted_ckpt_path",
        "verbose_evaluate",
    ],
)
def test_v2_0_0_unsupported_setters(attribute):
    trainer = Trainer()
    with pytest.raises(
        AttributeError, match=f"`Trainer.{attribute}` was deprecated in v1.6 and is no longer accessible as of v1.8."
    ):
        setattr(trainer, attribute, None)


def test_v2_0_0_unsupported_run_stage():
    trainer = Trainer()
    with pytest.raises(
        NotImplementedError, match="`Trainer.run_stage` was deprecated in v1.6 and is no longer supported as of v1.8."
    ):
        trainer.run_stage()


def test_v2_0_0_unsupported_call_hook():
    trainer = Trainer()
    with pytest.raises(
        NotImplementedError, match="`Trainer.call_hook` was deprecated in v1.6 and is no longer supported as of v1.8."
    ):
        trainer.call_hook("test_hook")


def test_v2_0_0_unsupported_data_loading_mixin():
    from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin

    class CustomTrainerDataLoadingMixin(TrainerDataLoadingMixin):
        pass

    with pytest.raises(
        NotImplementedError,
        match="`TrainerDataLoadingMixin` class was deprecated in v1.6 and is no longer supported as of v1.8",
    ):
        CustomTrainerDataLoadingMixin()

    trainer = Trainer()
    with pytest.raises(
        NotImplementedError,
        match="`Trainer.prepare_dataloader` was deprecated in v1.6 and is no longer supported as of v1.8.",
    ):
        trainer.prepare_dataloader(None)
    with pytest.raises(
        NotImplementedError,
        match="`Trainer.request_dataloader` was deprecated in v1.6 and is no longer supported as of v1.8.",
    ):
        trainer.request_dataloader(None)


def test_v2_0_0_trainer_optimizers_mixin():
    from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin

    class CustomTrainerOptimizersMixin(TrainerOptimizersMixin):
        pass

    with pytest.raises(
        NotImplementedError,
        match="`TrainerOptimizersMixin` class was deprecated in v1.6 and is no longer supported as of v1.8",
    ):
        CustomTrainerOptimizersMixin()

    trainer = Trainer()
    with pytest.raises(
        NotImplementedError,
        match="`Trainer.init_optimizers` was deprecated in v1.6 and is no longer supported as of v1.8.",
    ):
        trainer.init_optimizers(None)

    with pytest.raises(
        NotImplementedError,
        match="`Trainer.convert_to_lightning_optimizers` was deprecated in v1.6 and is no longer supported as of v1.8.",
    ):
        trainer.convert_to_lightning_optimizers()

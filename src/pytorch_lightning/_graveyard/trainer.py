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
import sys
from typing import Any, Optional

from pytorch_lightning import Trainer


def _patch_sys_modules() -> None:
    # TODO: Remove in v2.0.0
    self = sys.modules[__name__]
    sys.modules["pytorch_lightning.trainer.data_loading"] = self
    sys.modules["pytorch_lightning.trainer.optimizers"] = self


class TrainerDataLoadingMixin:
    # TODO: Remove in v2.0.0
    def __init__(self) -> None:
        raise NotImplementedError(
            "The `TrainerDataLoadingMixin` class was deprecated in v1.6 and is no longer supported as of v1.8."
        )


class TrainerOptimizersMixin:
    # TODO: Remove in v2.0.0
    def __init__(self) -> None:
        raise NotImplementedError(
            "The `TrainerOptimizersMixin` class was deprecated in v1.6 and is no longer supported as of v1.8."
        )


def _gpus(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.gpus` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` or `Trainer.device_ids` to get device information instead."
    )


def _root_gpu(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.root_gpu` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.strategy.root_device.index` instead."
    )


def _tpu_cores(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.tpu_cores` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` instead."
    )


def _ipus(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.ipus` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` instead."
    )


def _num_gpus(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.num_gpus` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` instead."
    )


def _devices(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.devices` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` or `Trainer.device_ids` to get device information instead."
    )


def _use_amp(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.use_amp` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.amp_backend`.",
    )


def _weights_save_path(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError("`Trainer.weights_save_path` was deprecated in v1.6 and is no longer accessible as of v1.8.")


def _lightning_optimizers(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.lightning_optimizers` was deprecated in v1.6 and is no longer accessible as of v1.8."
    )


def _should_rank_save_checkpoint(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.should_rank_save_checkpoint` was deprecated in v1.6 and is no longer accessible as of v1.8.",
    )


def _validated_ckpt_path(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "The `Trainer.validated_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _validated_ckpt_path_setter(_: Trainer, __: Optional[str]) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "The `Trainer.validated_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _tested_ckpt_path(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "The `Trainer.tested_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _tested_ckpt_path_setter(_: Trainer, __: Optional[str]) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "The `Trainer.tested_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _predicted_ckpt_path(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "The `Trainer.predicted_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _predicted_ckpt_path_setter(_: Trainer, __: Optional[str]) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "The `Trainer.predicted_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _verbose_evaluate(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "The `Trainer.verbose_evaluate` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `trainer.{validate,test}_loop.verbose` instead.",
    )


def _verbose_evaluate_setter(_: Trainer, __: bool) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "The `Trainer.verbose_evaluate` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `trainer.{validate,test}_loop.verbose` instead.",
    )


def _run_stage(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`Trainer.run_stage` was deprecated in v1.6 and is no longer supported as of v1.8."
        " Please use `Trainer.{fit,validate,test,predict}` instead."
    )


def _call_hook(_: Trainer, *__: Any, **___: Any) -> Any:
    # TODO: Remove in v2.0.0
    raise NotImplementedError("`Trainer.call_hook` was deprecated in v1.6 and is no longer supported as of v1.8.")


def _prepare_dataloader(_: Trainer, *__: Any, **___: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`Trainer.prepare_dataloader` was deprecated in v1.6 and is no longer supported as of v1.8."
    )


def _request_dataloader(_: Trainer, *__: Any, **___: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`Trainer.request_dataloader` was deprecated in v1.6 and is no longer supported as of v1.8."
    )


def _init_optimizers(_: Trainer, *__: Any, **___: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError("`Trainer.init_optimizers` was deprecated in v1.6 and is no longer supported as of v1.8.")


def _convert_to_lightning_optimizers(_: Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`Trainer.convert_to_lightning_optimizers` was deprecated in v1.6 and is no longer supported as of v1.8."
    )


def _reset_train_val_dataloaders(_: Trainer, *__: Any, **___: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`Trainer.reset_train_val_dataloaders` was deprecated in v1.7 and is no longer supported as of v1.9."
    )


_patch_sys_modules()

# Properties/Attributes
Trainer.gpus = property(_gpus)
Trainer.root_gpu = property(_root_gpu)
Trainer.tpu_cores = property(_tpu_cores)
Trainer.ipus = property(_ipus)
Trainer.num_gpus = property(_num_gpus)
Trainer.devices = property(_devices)
Trainer.use_amp = property(_use_amp)
Trainer.weights_save_path = property(_weights_save_path)
Trainer.lightning_optimizers = property(_lightning_optimizers)
Trainer.should_rank_save_checkpoint = property(_should_rank_save_checkpoint)
Trainer.validated_ckpt_path = property(fget=_validated_ckpt_path, fset=_validated_ckpt_path_setter)
Trainer.tested_ckpt_path = property(fget=_tested_ckpt_path, fset=_tested_ckpt_path_setter)
Trainer.predicted_ckpt_path = property(fget=_predicted_ckpt_path, fset=_predicted_ckpt_path_setter)
Trainer.verbose_evaluate = property(fget=_verbose_evaluate, fset=_verbose_evaluate_setter)


# Methods
Trainer.run_stage = _run_stage
Trainer.call_hook = _call_hook
Trainer.prepare_dataloader = _prepare_dataloader
Trainer.request_dataloader = _request_dataloader
Trainer.init_optimizers = _init_optimizers
Trainer.convert_to_lightning_optimizers = _convert_to_lightning_optimizers
Trainer.reset_train_val_dataloaders = _reset_train_val_dataloaders

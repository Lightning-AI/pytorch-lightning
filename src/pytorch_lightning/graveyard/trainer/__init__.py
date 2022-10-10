import sys
from typing import Any, Optional

from pytorch_lightning import Trainer

self = sys.modules[__name__]
sys.modules["pytorch_lightning.trainer"] = self
sys.modules["pytorch_lightning.trainer.trainer"] = self


def _gpus(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.gpus` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` or `Trainer.device_ids` to get device information instead."
    )


def _root_gpu(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.root_gpu` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.strategy.root_device.index` instead."
    )


def _tpu_cores(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.tpu_cores` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` instead."
    )


def _ipus(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.ipus` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` instead."
    )


def _num_gpus(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.num_gpus` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` instead."
    )


def _devices(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.devices` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.num_devices` or `Trainer.device_ids` to get device information instead."
    )


def _use_amp(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.use_amp` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.amp_backend`.",
    )


def _weights_save_path(_: Trainer) -> None:
    raise AttributeError("`Trainer.weights_save_path` was deprecated in v1.6 and is no longer accessible as of v1.8.")


def _lightning_optimizers(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.lightning_optimizers` was deprecated in v1.6 and is no longer accessible as of v1.8."
    )


def _should_rank_save_checkpoint(_: Trainer) -> None:
    raise AttributeError(
        "`Trainer.should_rank_save_checkpoint` was deprecated in v1.6 and is no longer accessible as of v1.8.",
    )


def _validated_ckpt_path(_: Trainer) -> None:
    raise AttributeError(
        "The `Trainer.validated_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _validated_ckpt_path_setter(_: Trainer, __: Optional[str]) -> None:
    raise AttributeError(
        "The `Trainer.validated_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _tested_ckpt_path(_: Trainer) -> None:
    raise AttributeError(
        "The `Trainer.tested_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _tested_ckpt_path_setter(_: Trainer, __: Optional[str]) -> None:
    raise AttributeError(
        "The `Trainer.tested_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _predicted_ckpt_path(_: Trainer) -> None:
    raise AttributeError(
        "The `Trainer.predicted_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _predicted_ckpt_path_setter(_: Trainer, __: Optional[str]) -> None:
    raise AttributeError(
        "The `Trainer.predicted_ckpt_path` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.ckpt_path` instead."
    )


def _verbose_evaluate(_: Trainer) -> None:
    raise AttributeError(
        "The `Trainer.verbose_evaluate` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `trainer.{validate,test}_loop.verbose` instead.",
    )


def _verbose_evaluate_setter(_: Trainer, __: bool) -> None:
    raise AttributeError(
        "The `Trainer.verbose_evaluate` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `trainer.{validate,test}_loop.verbose` instead.",
    )


def _run_stage(_: Trainer) -> None:
    raise NotImplementedError(
        "`Trainer.run_stage` was deprecated in v1.6 and is no longer supported as of v1.8."
        " Please use `Trainer.{fit,validate,test,predict}` instead."
    )


def _call_hook(_: Trainer, *__: Any, **___: Any) -> Any:
    raise NotImplementedError("`Trainer.call_hook` was deprecated in v1.6 and is no longer supported as of v1.8.")


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

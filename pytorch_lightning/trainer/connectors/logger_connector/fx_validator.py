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
from typing import Optional, Tuple, Union

from typing_extensions import TypedDict

from pytorch_lightning.utilities.exceptions import MisconfigurationException


class _FxValidator:
    class _LogOptions(TypedDict):
        allowed_on_step: Union[Tuple[bool], Tuple[bool, bool]]
        allowed_on_epoch: Union[Tuple[bool], Tuple[bool, bool]]
        default_on_step: bool
        default_on_epoch: bool

    functions = {
        "on_before_accelerator_backend_setup": None,
        "on_configure_sharded_model": None,
        "on_before_backward": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "backward": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "on_after_backward": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "on_before_optimizer_step": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "optimizer_step": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "lr_scheduler_step": None,
        "on_before_zero_grad": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "optimizer_zero_grad": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "on_init_start": None,
        "on_init_end": None,
        "on_fit_start": None,
        "on_fit_end": None,
        "on_sanity_check_start": None,
        "on_sanity_check_end": None,
        "on_train_start": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_train_end": None,
        "on_validation_start": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_validation_end": None,
        "on_test_start": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_test_end": None,
        "on_predict_start": None,
        "on_predict_end": None,
        "on_pretrain_routine_start": None,
        "on_pretrain_routine_end": None,
        "on_train_epoch_start": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_train_epoch_end": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_validation_epoch_start": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_validation_epoch_end": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_test_epoch_start": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_test_epoch_end": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_predict_epoch_start": None,
        "on_predict_epoch_end": None,
        "on_epoch_start": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_epoch_end": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "on_batch_start": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "on_batch_end": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "on_train_batch_start": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "on_train_batch_end": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "on_validation_batch_start": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=False, default_on_epoch=True
        ),
        "on_validation_batch_end": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=False, default_on_epoch=True
        ),
        "on_test_batch_start": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=False, default_on_epoch=True
        ),
        "on_test_batch_end": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=False, default_on_epoch=True
        ),
        "on_predict_batch_start": None,
        "on_predict_batch_end": None,
        "on_keyboard_interrupt": None,
        "on_exception": None,
        "state_dict": None,
        "on_save_checkpoint": None,
        "on_load_checkpoint": None,
        "load_state_dict": None,
        "setup": None,
        "teardown": None,
        "configure_sharded_model": None,
        "training_step": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "validation_step": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=False, default_on_epoch=True
        ),
        "test_step": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=False, default_on_epoch=True
        ),
        "predict_step": None,
        "training_step_end": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=True, default_on_epoch=False
        ),
        "validation_step_end": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=False, default_on_epoch=True
        ),
        "test_step_end": _LogOptions(
            allowed_on_step=(False, True), allowed_on_epoch=(False, True), default_on_step=False, default_on_epoch=True
        ),
        "training_epoch_end": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "validation_epoch_end": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "test_epoch_end": _LogOptions(
            allowed_on_step=(False,), allowed_on_epoch=(True,), default_on_step=False, default_on_epoch=True
        ),
        "configure_optimizers": None,
        "on_train_dataloader": None,
        "train_dataloader": None,
        "on_val_dataloader": None,
        "val_dataloader": None,
        "on_test_dataloader": None,
        "test_dataloader": None,
        "prepare_data": None,
        "configure_callbacks": None,
        "on_validation_model_eval": None,
        "on_test_model_eval": None,
        "on_validation_model_train": None,
        "on_test_model_train": None,
    }

    @classmethod
    def check_logging(cls, fx_name: str) -> None:
        """Check if the given hook is allowed to log."""
        if fx_name not in cls.functions:
            raise RuntimeError(
                f"Logging inside `{fx_name}` is not implemented."
                " Please, open an issue in `https://github.com/PyTorchLightning/pytorch-lightning/issues`."
            )

        if cls.functions[fx_name] is None:
            raise MisconfigurationException(f"You can't `self.log()` inside `{fx_name}`.")

    @classmethod
    def get_default_logging_levels(
        cls, fx_name: str, on_step: Optional[bool], on_epoch: Optional[bool]
    ) -> Tuple[bool, bool]:
        """Return default logging levels for given hook."""
        fx_config = cls.functions[fx_name]
        assert fx_config is not None
        on_step = fx_config["default_on_step"] if on_step is None else on_step
        on_epoch = fx_config["default_on_epoch"] if on_epoch is None else on_epoch
        return on_step, on_epoch

    @classmethod
    def check_logging_levels(cls, fx_name: str, on_step: bool, on_epoch: bool) -> None:
        """Check if the logging levels are allowed in the given hook."""
        fx_config = cls.functions[fx_name]
        assert fx_config is not None
        m = "You can't `self.log({}={})` inside `{}`, must be one of {}."
        if on_step not in fx_config["allowed_on_step"]:
            msg = m.format("on_step", on_step, fx_name, fx_config["allowed_on_step"])
            raise MisconfigurationException(msg)

        if on_epoch not in fx_config["allowed_on_epoch"]:
            msg = m.format("on_epoch", on_epoch, fx_name, fx_config["allowed_on_epoch"])
            raise MisconfigurationException(msg)

    @classmethod
    def check_logging_and_get_default_levels(
        cls, fx_name: str, on_step: Optional[bool], on_epoch: Optional[bool]
    ) -> Tuple[bool, bool]:
        """Check if the given hook name is allowed to log and return logging levels."""
        cls.check_logging(fx_name)
        on_step, on_epoch = cls.get_default_logging_levels(fx_name, on_step, on_epoch)
        cls.check_logging_levels(fx_name, on_step, on_epoch)
        return on_step, on_epoch

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
from typing import Tuple, Union

from typing_extensions import TypedDict

from pytorch_lightning.utilities.exceptions import MisconfigurationException


class _FxValidator:
    class _LogOptions(TypedDict):
        on_step: Union[Tuple[bool], Tuple[bool, bool]]
        on_epoch: Union[Tuple[bool], Tuple[bool, bool]]

    functions = {
        "on_before_accelerator_backend_setup": None,
        "on_configure_sharded_model": None,
        "on_before_backward": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_after_backward": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_before_optimizer_step": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_before_zero_grad": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_init_start": None,
        "on_init_end": None,
        "on_fit_start": None,
        "on_fit_end": None,
        "on_sanity_check_start": None,
        "on_sanity_check_end": None,
        "on_train_start": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_train_end": None,
        "on_validation_start": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_validation_end": None,
        "on_test_start": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_test_end": None,
        "on_predict_start": None,
        "on_predict_end": None,
        "on_pretrain_routine_start": None,
        "on_pretrain_routine_end": None,
        "on_train_epoch_start": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_train_epoch_end": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_validation_epoch_start": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_validation_epoch_end": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_test_epoch_start": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_test_epoch_end": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_predict_epoch_start": None,
        "on_predict_epoch_end": None,
        "on_epoch_start": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_epoch_end": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "on_batch_start": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_batch_end": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_train_batch_start": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_train_batch_end": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_validation_batch_start": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_validation_batch_end": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_test_batch_start": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_test_batch_end": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "on_predict_batch_start": None,
        "on_predict_batch_end": None,
        "on_keyboard_interrupt": None,
        "on_exception": None,
        "on_save_checkpoint": None,
        "on_load_checkpoint": None,
        "setup": None,
        "teardown": None,
        "configure_sharded_model": None,
        "training_step": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "validation_step": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "test_step": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "predict_step": None,
        "training_step_end": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "validation_step_end": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "test_step_end": _LogOptions(on_step=(False, True), on_epoch=(False, True)),
        "training_epoch_end": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "validation_epoch_end": _LogOptions(on_step=(False,), on_epoch=(True,)),
        "test_epoch_end": _LogOptions(on_step=(False,), on_epoch=(True,)),
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
    }

    @classmethod
    def check_logging(cls, fx_name: str, on_step: bool, on_epoch: bool) -> None:
        """Check if the given function name is allowed to log."""
        if fx_name not in cls.functions:
            raise RuntimeError(
                f"Logging inside `{fx_name}` is not implemented."
                " Please, open an issue in `https://github.com/PyTorchLightning/pytorch-lightning/issues`"
            )
        allowed = cls.functions[fx_name]
        if allowed is None:
            raise MisconfigurationException(f"You can't `self.log()` inside `{fx_name}`")

        m = "You can't `self.log({}={})` inside `{}`, must be one of {}"
        if on_step not in allowed["on_step"]:
            msg = m.format("on_step", on_step, fx_name, allowed["on_step"])
            raise MisconfigurationException(msg)

        if on_epoch not in allowed["on_epoch"]:
            msg = m.format("on_epoch", on_epoch, fx_name, allowed["on_epoch"])
            raise MisconfigurationException(msg)

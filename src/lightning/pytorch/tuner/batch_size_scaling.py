# Copyright The Lightning AI team.
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
# limitations under the License
import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.memory import garbage_collection_cuda, is_oom_error
from lightning.pytorch.utilities.parsing import lightning_getattr, lightning_setattr
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

log = logging.getLogger(__name__)


def _scale_batch_size(
    trainer: "pl.Trainer",
    mode: str = "power",
    steps_per_trial: int = 3,
    init_val: int = 2,
    max_trials: int = 25,
    batch_arg_name: str = "batch_size",
) -> Optional[int]:
    """Iteratively try to find the largest batch size for a given model that does not give an out of memory (OOM)
    error.

    Args:
        trainer: A Trainer instance.
        mode: Search strategy to update the batch size:

            - ``'power'``: Keep multiplying the batch size by 2, until we get an OOM error.
            - ``'binsearch'``: Initially keep multiplying by 2 and after encountering an OOM error
                do a binary search between the last successful batch size and the batch size that failed.

        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs,
            however in practise a few are needed
        init_val: initial batch size to start the search with
        max_trials: max number of increases in batch size done before
           algorithm is terminated
        batch_arg_name: name of the attribute that stores the batch size.
            It is expected that the user has provided a model or datamodule that has a hyperparameter
            with that name. We will look for this attribute name in the following places

            - ``model``
            - ``model.hparams``
            - ``trainer.datamodule`` (the datamodule passed to the tune method)

    """
    if trainer.fast_dev_run:
        rank_zero_warn("Skipping batch size scaler since `fast_dev_run` is enabled.")
        return None

    # Save initial model, that is loaded after batch size is found
    ckpt_path = os.path.join(trainer.default_root_dir, f".scale_batch_size_{uuid.uuid4()}.ckpt")
    trainer.save_checkpoint(ckpt_path)

    # Arguments we adjust during the batch size finder, save for restoring
    params = __scale_batch_dump_params(trainer)

    # Set to values that are required by the algorithm
    __scale_batch_reset_params(trainer, steps_per_trial)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()

    new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=init_val)

    if mode == "power":
        new_size = _run_power_scaling(trainer, new_size, batch_arg_name, max_trials, params)
    elif mode == "binsearch":
        new_size = _run_binary_scaling(trainer, new_size, batch_arg_name, max_trials, params)

    garbage_collection_cuda()

    log.info(f"Finished batch size finder, will continue with full run using batch size {new_size}")

    __scale_batch_restore_params(trainer, params)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()

    trainer._checkpoint_connector.restore(ckpt_path)
    trainer.strategy.remove_checkpoint(ckpt_path)

    return new_size


def __scale_batch_dump_params(trainer: "pl.Trainer") -> dict[str, Any]:
    dumped_params = {
        "loggers": trainer.loggers,
        "callbacks": trainer.callbacks,
    }
    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        dumped_params["max_steps"] = trainer.max_steps
        dumped_params["limit_train_batches"] = trainer.limit_train_batches
        dumped_params["limit_val_batches"] = trainer.limit_val_batches
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        dumped_params["limit_eval_batches"] = getattr(trainer, f"limit_{stage.dataloader_prefix}_batches")
        dumped_params["loop_verbose"] = loop.verbose

    dumped_params["loop_state_dict"] = deepcopy(loop.state_dict())
    return dumped_params


def __scale_batch_reset_params(trainer: "pl.Trainer", steps_per_trial: int) -> None:
    from lightning.pytorch.loggers.logger import DummyLogger

    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.callbacks = []

    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        trainer.limit_train_batches = 1.0
        trainer.limit_val_batches = steps_per_trial
        trainer.fit_loop.epoch_loop.max_steps = steps_per_trial
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        setattr(trainer, f"limit_{stage.dataloader_prefix}_batches", steps_per_trial)
        loop.verbose = False


def __scale_batch_restore_params(trainer: "pl.Trainer", params: dict[str, Any]) -> None:
    # TODO: There are more states that needs to be reset (#4512 and #4870)
    trainer.loggers = params["loggers"]
    trainer.callbacks = params["callbacks"]

    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        loop.epoch_loop.max_steps = params["max_steps"]
        trainer.limit_train_batches = params["limit_train_batches"]
        trainer.limit_val_batches = params["limit_val_batches"]
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        setattr(trainer, f"limit_{stage.dataloader_prefix}_batches", params["limit_eval_batches"])

    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    if isinstance(loop, pl.loops._EvaluationLoop) and "loop_verbose" in params:
        loop.verbose = params["loop_verbose"]

    # make sure the loop's state is reset
    _reset_dataloaders(trainer)
    loop.reset()


def _run_power_scaling(
    trainer: "pl.Trainer",
    new_size: int,
    batch_arg_name: str,
    max_trials: int,
    params: dict[str, Any],
) -> int:
    """Batch scaling mode where the size is doubled at each iteration until an OOM error is encountered."""
    # this flag is used to determine whether the previously scaled batch size, right before OOM, was a success or not
    # if it was we exit, else we continue downscaling in case we haven't encountered a single optimal batch size
    any_success = False
    for _ in range(max_trials):
        garbage_collection_cuda()

        # reset after each try
        _reset_progress(trainer)

        try:
            _try_loop_run(trainer, params)
            new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc="succeeded")

            if not changed:
                break

            # Force the train dataloader to reset as the batch size has changed
            _reset_dataloaders(trainer)
            any_success = True
        except RuntimeError as exception:
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, factor=0.5, desc="failed")
                # Force the train dataloader to reset as the batch size has changed
                _reset_dataloaders(trainer)
                if any_success:
                    break
            else:
                raise  # some other error not memory related

    return new_size


def _run_binary_scaling(
    trainer: "pl.Trainer",
    new_size: int,
    batch_arg_name: str,
    max_trials: int,
    params: dict[str, Any],
) -> int:
    """Batch scaling mode where the size is initially is doubled at each iteration until an OOM error is encountered.

    Hereafter, the batch size is further refined using a binary search

    """
    low = 1
    high = None
    count = 0
    while True:
        garbage_collection_cuda()

        # reset after each try
        _reset_progress(trainer)

        try:
            # run loop
            _try_loop_run(trainer, params)
            count += 1
            if count > max_trials:
                break
            # Double in size
            low = new_size
            if high:
                if high - low <= 1:
                    break
                midval = (high + low) // 2
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc="succeeded")
            else:
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc="succeeded")

            if not changed:
                break

            # Force the train dataloader to reset as the batch size has changed
            _reset_dataloaders(trainer)

        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()

                high = new_size
                midval = (high + low) // 2
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc="failed")

                # Force the train dataloader to reset as the batch size has changed
                _reset_dataloaders(trainer)

                if high - low <= 1:
                    break
            else:
                raise  # some other error not memory related

    return new_size


def _adjust_batch_size(
    trainer: "pl.Trainer",
    batch_arg_name: str = "batch_size",
    factor: float = 1.0,
    value: Optional[int] = None,
    desc: Optional[str] = None,
) -> tuple[int, bool]:
    """Helper function for adjusting the batch size.

    Args:
        trainer: instance of lightning.pytorch.Trainer
        factor: value which the old batch size is multiplied by to get the
            new batch size
        value: if a value is given, will override the batch size with this value.
            Note that the value of `factor` will not have an effect in this case
        desc: either ``"succeeded"`` or ``"failed"``. Used purely for logging

    Returns:
        The new batch size for the next trial and a bool that signals whether the
        new value is different than the previous batch size.

    """
    model = trainer.lightning_module
    batch_size = lightning_getattr(model, batch_arg_name)
    assert batch_size is not None

    loop = trainer._active_loop
    assert loop is not None
    loop.setup_data()
    combined_loader = loop._combined_loader
    assert combined_loader is not None
    try:
        combined_dataset_length = combined_loader._dataset_length()
        if batch_size >= combined_dataset_length:
            rank_zero_info(f"The batch size {batch_size} is greater or equal than the length of your dataset.")
            return batch_size, False
    except NotImplementedError:
        # all datasets are iterable style
        pass

    new_size = value if value is not None else int(batch_size * factor)
    if desc:
        rank_zero_info(f"Batch size {batch_size} {desc}, trying batch size {new_size}")
    changed = new_size != batch_size
    lightning_setattr(model, batch_arg_name, new_size)
    return new_size, changed


def _reset_dataloaders(trainer: "pl.Trainer") -> None:
    loop = trainer._active_loop
    assert loop is not None
    loop._combined_loader = None  # force a reload
    loop.setup_data()
    if isinstance(loop, pl.loops._FitLoop):
        loop.epoch_loop.val_loop._combined_loader = None
        loop.epoch_loop.val_loop.setup_data()


def _try_loop_run(trainer: "pl.Trainer", params: dict[str, Any]) -> None:
    loop = trainer._active_loop
    assert loop is not None
    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    loop.run()


def _reset_progress(trainer: "pl.Trainer") -> None:
    if trainer.lightning_module.automatic_optimization:
        trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.reset()
    else:
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.reset()

    trainer.fit_loop.epoch_progress.reset()

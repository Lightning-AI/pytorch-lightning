# Copyright Lightning AI.
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
import traceback
from functools import partial
from typing import Any, Callable

from lightning_lite.utilities.distributed import distributed_available
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.trainer.states import TrainerStatus
from pytorch_lightning.utilities.exceptions import _TunerExitException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn


def call_and_handle_interrupt(trainer: Any, trainer_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    r"""
    Error handling, intended to be used only for main trainer function entry points (fit, validate, test, predict)
    as all errors should funnel through them

    Args:
        trainer_fn: one of (fit, validate, test, predict)
        *args: positional arguments to be passed to the `trainer_fn`
        **kwargs: keyword arguments to be passed to `trainer_fn`
    """
    try:
        strategy = trainer.strategy
        run_method = partial(_wrap_run_method, trainer_fn, strategy)
        if strategy.launcher is not None:
            return strategy.launcher.launch(run_method, *args, trainer=trainer, **kwargs)
        else:
            return run_method(*args, **kwargs)

    except _TunerExitException:
        trainer._call_teardown_hook()
        trainer._teardown()
        trainer.state.status = TrainerStatus.FINISHED
        trainer.state.stage = None

    # TODO: Unify both exceptions below, where `KeyboardError` doesn't re-raise
    except KeyboardInterrupt as exception:
        rank_zero_warn("Detected KeyboardInterrupt, attempting graceful shutdown...")
        # user could press Ctrl+c many times... only shutdown once
        if not trainer.interrupted:
            trainer.state.status = TrainerStatus.INTERRUPTED
            trainer._call_callback_hooks("on_exception", exception)
            for logger in trainer.loggers:
                logger.finalize("failed")
    except BaseException as exception:
        trainer.state.status = TrainerStatus.INTERRUPTED
        if distributed_available() and trainer.world_size > 1:
            # try syncing remaining processes, kill otherwise
            strategy.reconciliate_processes(traceback.format_exc())
        trainer._call_callback_hooks("on_exception", exception)
        for logger in trainer.loggers:
            logger.finalize("failed")
        trainer._teardown()
        # teardown might access the stage so we reset it after
        trainer.state.stage = None
        raise


def _wrap_run_method(run_method: Callable, strategy: Strategy, *args: Any, **kwargs: Any) -> Any:
    if not hasattr(strategy.launcher, "_strategy"):
        # the launcher (if set) should manage this if it has access to the strategy. call this otherwise
        strategy.setup_environment()
    return run_method(*args, **kwargs)

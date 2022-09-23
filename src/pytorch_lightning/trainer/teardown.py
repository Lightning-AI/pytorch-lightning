import traceback
from typing import Any, Callable
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.trainer.states import TrainerStatus
from lightning_lite.utilities.distributed import distributed_available


def call_and_handle_interrupt(trainer, trainer_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    r"""
    Error handling, intended to be used only for main trainer function entry points (fit, validate, test, predict)
    as all errors should funnel through them

    Args:
        trainer_fn: one of (fit, validate, test, predict)
        *args: positional arguments to be passed to the `trainer_fn`
        **kwargs: keyword arguments to be passed to `trainer_fn`
    """
    try:
        if trainer.strategy.launcher is not None:
            return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
        else:
            return trainer_fn(*args, **kwargs)
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
            trainer.strategy.reconciliate_processes(traceback.format_exc())
        trainer._call_callback_hooks("on_exception", exception)
        for logger in trainer.loggers:
            logger.finalize("failed")
        trainer._teardown()
        # teardown might access the stage so we reset it after
        trainer.state.stage = None
        raise


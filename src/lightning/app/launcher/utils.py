import functools
import logging
import os
import signal
from typing import Any, Callable

import psutil
from lightning_cloud.openapi import V1LightningworkState

from lightning.app import LightningApp, _logger, _root_logger
from lightning.app import __version__ as LIGHTNING_VERSION
from lightning.app.utilities.enum import WorkStageStatus


def cloud_work_stage_to_work_status_stage(stage: V1LightningworkState) -> str:
    """Maps the Work stage names from the Grid cloud backend to the status names in the Lightning framework."""
    mapping = {
        V1LightningworkState.STOPPED: WorkStageStatus.STOPPED,
        V1LightningworkState.PENDING: WorkStageStatus.PENDING,
        V1LightningworkState.NOT_STARTED: WorkStageStatus.PENDING,
        V1LightningworkState.IMAGE_BUILDING: WorkStageStatus.PENDING,
        V1LightningworkState.RUNNING: WorkStageStatus.RUNNING,
        V1LightningworkState.FAILED: WorkStageStatus.FAILED,
    }
    if stage not in mapping:
        raise ValueError(f"Cannot map the lightning-cloud work state {stage} to the lightning status stage.")
    return mapping[stage]


def _print_to_logger_info(*args, **kwargs):
    # TODO Find a better way to re-direct print to loggers.
    _logger.info(" ".join([str(v) for v in args]))


def convert_print_to_logger_info(func: Callable) -> Callable:
    """This function is used to transform any print into logger.info calls, so it gets tracked in the cloud."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        original_print = __builtins__["print"]
        __builtins__["print"] = _print_to_logger_info
        res = func(*args, **kwargs)
        __builtins__["print"] = original_print
        return res

    return wrapper


def _enable_debugging():
    tar_file = os.path.join(os.getcwd(), f"lightning-{LIGHTNING_VERSION}.tar.gz")

    if not os.path.exists(tar_file):
        return

    _root_logger.propagate = True
    _logger.propagate = True
    _root_logger.setLevel(logging.DEBUG)
    _root_logger.debug("Setting debugging mode.")


def enable_debugging(func: Callable) -> Callable:
    """This function is used set the logging level to DEBUG and set it back to INFO once the function is done."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _enable_debugging()
        res = func(*args, **kwargs)
        _logger.setLevel(logging.INFO)
        return res

    return wrapper


def exit_app(app: LightningApp):
    """This function checks if dumb-init is running on process 0 and exits the containter with exit code 0.

    Otherwise we fall back to stopping the app via backend API call

    """
    try:
        # Get process information for PID 1, where dumb-init is running
        process = psutil.Process(1)
        process_name = process.name()

        # This kills the dumb-init process running on pid 1
        # There's issues propagating the exit code through regular python
        # program exit, so we directly kill the dumb-init process
        # which causes the flow container to exit with status code 0
        if "dumb-init" in process_name.lower():
            print("Killing dumb-init and exiting the container..")
            os.kill(1, signal.SIGTERM)
        else:
            print("Process 1 not running dumb-init. Stopping the app..")
            app.backend.stop_app(app)
    except psutil.NoSuchProcess:
        print("Process with PID 1 not found. Stopping the app..")
        app.backend.stop_app(app)

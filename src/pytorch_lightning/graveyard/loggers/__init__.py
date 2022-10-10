import sys

from pytorch_lightning.loggers import *  # noqa: F403
from pytorch_lightning.loggers.base import *  # noqa: F403
from pytorch_lightning.loggers.logger import *  # noqa: F403

self = sys.modules[__name__]
sys.modules["pytorch_lightning.loggers"] = self
sys.modules["pytorch_lightning.loggers.logger"] = self
sys.modules["pytorch_lightning.loggers.base"] = self


class LoggerCollection:
    def __init__(self, _: Any):
        raise RuntimeError(
            "`LoggerCollection` was deprecated in v1.6 and removed in v1.8. Directly pass a list of loggers"
            " to the Trainer and access the list via the `trainer.loggers` attribute."
        )


def _update_agg_funcs(logger: Logger, *__: Any, **___: Any) -> None:
    raise NotImplementedError(
        f"`{type(logger).__name__}.update_agg_funcs` was deprecated in v1.6 and is no longer supported as of v1.8."
    )


def _agg_and_log_metrics(logger: Logger, *__: Any, **___: Any) -> None:
    raise NotImplementedError(
        f"`{type(logger).__name__}.update_agg_funcs` was deprecated in v1.6 and is no longer supported as of v1.8."
    )


# Methods
Logger.update_agg_funcs = _update_agg_funcs
Logger.agg_and_log_metrics = _agg_and_log_metrics

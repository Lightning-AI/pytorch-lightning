import argparse
from abc import ABC
from functools import wraps
from typing import Union, Optional, Dict, Iterable, Any, Callable, List


def rank_zero_only(fn: Callable):
    """Decorate a logger method to run it only on the process with rank 0.

    Args:
        fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.rank == 0:
            fn(self, *args, **kwargs)

    return wrapped_fn


class LightningLoggerBase(ABC):
    """Base class for experiment loggers."""

    def __init__(self):
        self._rank = 0

    @property
    def experiment(self) -> Any:
        raise NotImplementedError()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Record metrics.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        raise NotImplementedError()

    def log_hyperparams(self, params: argparse.Namespace):
        """Record hyperparameters.

        Args:
            params: argparse.Namespace containing the hyperparameters
        """
        raise NotImplementedError()

    def save(self):
        """Save log data."""

    def finalize(self, status: str):
        """Do any processing that is necessary to finalize an experiment.

        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """

    def close(self):
        """Do any cleanup that is necessary to close an experiment."""

    @property
    def rank(self) -> int:
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        return self._rank

    @rank.setter
    def rank(self, value: int):
        """Set the process rank."""
        self._rank = value

    @property
    def name(self) -> str:
        """Return the experiment name."""
        raise NotImplementedError("Sub-classes must provide a name property")

    @property
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        raise NotImplementedError("Sub-classes must provide a version property")


class LoggerCollection(LightningLoggerBase):
    """The `LoggerCollection` class is used to iterate all logging actions over the given `logger_iterable`.

    Args:
        logger_iterable: An iterable collection of loggers
    """

    def __init__(self, logger_iterable: Iterable[LightningLoggerBase]):
        super().__init__()
        self._logger_iterable = logger_iterable

    @property
    def experiment(self) -> List[Any]:
        return [logger.experiment() for logger in self._logger_iterable]

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        [logger.log_metrics(metrics, step) for logger in self._logger_iterable]

    def log_hyperparams(self, params: argparse.Namespace):
        [logger.log_hyperparams(params) for logger in self._logger_iterable]

    def save(self):
        [logger.save() for logger in self._logger_iterable]

    def finalize(self, status: str):
        [logger.finalize(status) for logger in self._logger_iterable]

    def close(self):
        [logger.close() for logger in self._logger_iterable]

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, value: int):
        self._rank = value
        for logger in self._logger_iterable:
            logger.rank = value

    @property
    def name(self) -> str:
        return '_'.join([str(logger.name) for logger in self._logger_iterable])

    @property
    def version(self) -> str:
        return '_'.join([str(logger.version) for logger in self._logger_iterable])

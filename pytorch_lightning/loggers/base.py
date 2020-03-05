import argparse
from abc import ABC, abstractmethod
from argparse import Namespace
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
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this logger"""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Record metrics.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        pass

    def _convert_params(self, params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @abstractmethod
    def log_hyperparams(self, params: argparse.Namespace):
        """Record hyperparameters.

        Args:
            params: argparse.Namespace containing the hyperparameters
        """
        pass

    def save(self) -> None:
        """Save log data."""
        pass

    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment.

        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        pass

    def close(self) -> None:
        """Do any cleanup that is necessary to close an experiment."""
        pass

    @property
    def rank(self) -> int:
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        return self._rank

    @rank.setter
    def rank(self, value: int) -> None:
        """Set the process rank."""
        self._rank = value

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the experiment name."""
        pass

    @property
    @abstractmethod
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        pass


class LoggerCollection(LightningLoggerBase):
    """The `LoggerCollection` class is used to iterate all logging actions over the given `logger_iterable`.

    Args:
        logger_iterable: An iterable collection of loggers
    """

    def __init__(self, logger_iterable: Iterable[LightningLoggerBase]):
        super().__init__()
        self._logger_iterable = logger_iterable

    def __getitem__(self, index: int) -> LightningLoggerBase:
        return [logger for logger in self._logger_iterable][index]

    @property
    def experiment(self) -> List[Any]:
        return [logger.experiment for logger in self._logger_iterable]

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        [logger.log_metrics(metrics, step) for logger in self._logger_iterable]

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        [logger.log_hyperparams(params) for logger in self._logger_iterable]

    def save(self) -> None:
        [logger.save() for logger in self._logger_iterable]

    def finalize(self, status: str) -> None:
        [logger.finalize(status) for logger in self._logger_iterable]

    def close(self) -> None:
        [logger.close() for logger in self._logger_iterable]

    @LightningLoggerBase.rank.setter
    def rank(self, value: int) -> None:
        self._rank = value
        for logger in self._logger_iterable:
            logger.rank = value

    @property
    def name(self) -> str:
        return '_'.join([str(logger.name) for logger in self._logger_iterable])

    @property
    def version(self) -> str:
        return '_'.join([str(logger.version) for logger in self._logger_iterable])

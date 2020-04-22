"""
Test Tube
---------
"""
from argparse import Namespace
from typing import Optional, Dict, Any, Union

try:
    from test_tube import Experiment
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `test_tube` logger which is not installed yet,'  # pragma: no-cover
                      ' install it with `pip install test-tube`.')

from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only


class TestTubeLogger(LightningLoggerBase):
    r"""
    Log to local file system in `TensorBoard <https://www.tensorflow.org/tensorboard>`_ format
    but using a nicer folder structure (see `full docs <https://williamfalcon.github.io/test-tube>`_).
    Install it with pip:

    .. code-block:: bash

        pip install test_tube

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import TestTubeLogger
        >>> logger = TestTubeLogger("tt_logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    >>> from pytorch_lightning import LightningModule
    >>> class LitModel(LightningModule):
    ...     def training_step(self, batch, batch_idx):
    ...         # example
    ...         self.logger.experiment.whatever_method_summary_writer_supports(...)
    ...
    ...     def any_lightning_module_function_or_hook(self):
    ...         self.logger.experiment.add_histogram(...)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        description: A short snippet about this experiment
        debug: If ``True``, it doesn't log anything.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        create_git_tag: If ``True`` creates a git tag to save the code used in this experiment.

    """

    __test__ = False

    def __init__(self,
                 save_dir: str,
                 name: str = "default",
                 description: Optional[str] = None,
                 debug: bool = False,
                 version: Optional[int] = None,
                 create_git_tag: bool = False):
        super().__init__()
        self.save_dir = save_dir
        self._name = name
        self.description = description
        self.debug = debug
        self._version = version
        self.create_git_tag = create_git_tag
        self._experiment = None

    @property
    def experiment(self) -> Experiment:
        r"""

        Actual TestTube object. To use TestTube features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_test_tube_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._experiment = Experiment(
            save_dir=self.save_dir,
            name=self._name,
            debug=self.debug,
            version=self.version,
            description=self.description,
            create_git_tag=self.create_git_tag,
            rank=rank_zero_only.rank,
        )
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        self.experiment.argparse(Namespace(**params))

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.experiment.log(metrics, global_step=step)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.save()
        self.close()

    @rank_zero_only
    def close(self) -> None:
        super().save()
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        if not self.debug:
            exp = self.experiment
            exp.close()

    @property
    def name(self) -> str:
        if self._experiment is None:
            return self._name
        else:
            return self.experiment.name

    @property
    def version(self) -> int:
        if self._experiment is None:
            return self._version
        else:
            return self.experiment.version

    # Test tube experiments are not pickleable, so we need to override a few
    # methods to get DDP working. See
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    # for more info.
    def __getstate__(self) -> Dict[Any, Any]:
        state = self.__dict__.copy()
        state["_experiment"] = self.experiment.get_meta_copy()
        return state

    def __setstate__(self, state: Dict[Any, Any]):
        self._experiment = state["_experiment"].get_non_ddp_exp()
        del state["_experiment"]
        self.__dict__.update(state)

"""
Log using `sacred <https://sacred.readthedocs.io/en/stable/index.html>'_
.. code-block:: python
    from pytorch_lightning.logging import SacredLogger
    ex = Experiment() # initialize however you like
    ex.main(your_main_fct)
    ex.observers.append(
        # add any observer you like
    )
    sacred_logger = SacredLogger(ex)
    trainer = Trainer(logger=sacred_logger)
Use the logger anywhere in you LightningModule as follows:
.. code-block:: python
    def train_step(...):
        # example
        self.logger.experiment.whatever_sacred_supports(...)
    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.whatever_sacred_supports(...)
"""

from logging import getLogger
from time import time

try:
    import sacred
except ImportError:
    raise ImportError('Missing sacred package.  Run `pip install sacred`')

from pytorch_lightning.logging.base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


# TODO: add docstring with type definition
class SacredLogger(LightningLoggerBase):
    def __init__(self, sacred_experiment):
        """Initialize a sacred logger.

        :param sacred.experiment.Experiment sacred_experiment: Required. Experiment object with desired observers
        already appended.
        """
        super().__init__()
        self.sacred_experiment = sacred_experiment
        self.experiment_name = sacred_experiment.path
        self._run_id = None

    @property
    def experiment(self):
        return self.sacred_experiment

    @property
    def run_id(self):
        if self._run_id is not None:
            return self._run_id

        self._run_id = self.sacred_experiment.current_run._id
        return self._run_id

    @rank_zero_only
    def log_hyperparams(self, params):
        # probably not needed bc. it is dealt with by sacred
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(
                    f"Discarding metric with string value {k}={v}"
                )
                continue
            self.experiment.log_scalar(k, v, step)

    @property
    def name(self):
        return self.experiment_name

    @property
    def version(self):
        return self.run_id

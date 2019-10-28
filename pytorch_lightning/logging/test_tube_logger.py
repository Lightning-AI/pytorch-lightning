try:
    from test_tube import Experiment
except ImportError:
    raise ImportError('Missing test-tube package.')

from .base import LightningLoggerBase, rank_zero_only


class TestTubeLogger(LightningLoggerBase):
    __test__ = False

    def __init__(
            self, save_dir, name="default", description=None, debug=False,
            version=None, create_git_tag=False
    ):
        super().__init__()
        self.save_dir = save_dir
        self.name = name
        self.description = description
        self.debug = debug
        self._version = version
        self.create_git_tag = create_git_tag
        self._experiment = None

    @property
    def experiment(self):
        if self._experiment is not None:
            return self._experiment

        self._experiment = Experiment(
            save_dir=self.save_dir,
            name=self.name,
            debug=self.debug,
            version=self.version,
            description=self.description,
            create_git_tag=self.create_git_tag,
            rank=self.rank,
        )
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.experiment.argparse(params)

    @rank_zero_only
    def log_metrics(self, metrics, step_num=None):
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.experiment.log(metrics, global_step=step_num)

    @rank_zero_only
    def save(self):
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status):
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.save()
        self.close()

    @rank_zero_only
    def close(self):
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        exp = self.experiment
        exp.close()

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value
        if self._experiment is not None:
            self.experiment.rank = value

    @property
    def version(self):
        if self._experiment is None:
            return self._version
        else:
            return self.experiment.version

    # Test tube experiments are not pickleable, so we need to override a few
    # methods to get DDP working. See
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    # for more info.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = self.experiment.get_meta_copy()
        return state

    def __setstate__(self, state):
        self._experiment = state["_experiment"].get_non_ddp_exp()
        del state["_experiment"]
        self.__dict__.update(state)

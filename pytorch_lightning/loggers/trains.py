"""
Log using `allegro.ai TRAINS <https://github.com/allegroai/trains>'_

.. code-block:: python

    from pytorch_lightning.loggers import TrainsLogger
    trains_logger = TrainsLogger(
        project_name="pytorch lightning",
        task_name="default",
    )
    trainer = Trainer(logger=trains_logger)


Use the logger anywhere in you LightningModule as follows:

.. code-block:: python

    def train_step(...):
        # example
        self.logger.experiment.whatever_trains_supports(...)

    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.whatever_trains_supports(...)

"""

from logging import getLogger
import torch

try:
    import trains
except ImportError:
    raise ImportError('Missing TRAINS package.')

from .base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


class TrainsLogger(LightningLoggerBase):
    def __init__(self, project_name=None, task_name=None, **kwargs):
        r"""

        Logs using TRAINS

        Args:
            project_name (str): The name of the experiment's project
            task_name (str): The name of the experiment
        """
        super().__init__()
        self._trains = trains.Task.init(project_name=project_name, task_name=task_name, **kwargs)

    @property
    def experiment(self):
        r"""

        Actual TRAINS object. To use TRAINS features do the following.

        Example::

            self.logger.experiment.some_trains_function()

        """
        return self._trains

    @property
    def id(self):
        if not self._trains:
            return None
        return self._trains.id

    @rank_zero_only
    def log_hyperparams(self, params):
        if not self._trains:
            return None
        if not params:
            return
        if isinstance(params, dict):
            self._trains.connect(params)
        else:
            self._trains.connect(vars(params))

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if not self._trains:
            return None
        if not step:
            step = self._trains.get_last_iteration()
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning("Discarding metric with string value {}={}".format(k, v))
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            parts = k.split('/')
            if len(parts) <= 1:
                series = title = k
            else:
                title = parts[0]
                series = parts[1:]
            self._trains.get_logger().report_scalar(title=title, series=series, value=v, iteration=step)

    @rank_zero_only
    def log_metric(self, title, series, value, step=None):
        """Log metrics (numeric values) in TRAINS experiments

        :param str title: The title of the graph to log, e.g. loss, accuracy.
        :param str series: The series name in the graph, e.g. classification, localization
        :param float value: The value to log
        :param int|None step: Step number at which the metrics should be recorded
        """
        if not self._trains:
            return None
        if not step:
            step = self._trains.get_last_iteration()
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._trains.get_logger().report_scalar(title=title, series=series, value=value, iteration=step)

    @rank_zero_only
    def log_text(self, text):
        """Log console text data in TRAINS experiment

        :param str text: The value of the log (data-point).
        """
        if not self._trains:
            return None
        self._trains.get_logger().report_text(text)

    @rank_zero_only
    def log_image(self, title, series, image, step=None):
        """Log Debug image in TRAINS experiment

        :param str title: The title of the debug image, i.e. "failed", "passed".
        :param str series: The series name of the debug image, i.e. "Image 0", "Image 1".
        :param str|Numpy|PIL.Image image: Debug image to log.
           Can be one of the following types: Numpy, PIL image, path to image file (str)
        :param int|None step: Step number at which the metrics should be recorded
        """
        if not self._trains:
            return None
        if not step:
            step = self._trains.get_last_iteration()
        if isinstance(image, str):
            self._trains.get_logger().report_image(title=title, series=series, local_path=image, iteration=step)
        else:
            self._trains.get_logger().report_image(title=title, series=series, image=image, iteration=step)

    @rank_zero_only
    def log_artifact(self, name, artifact, metadata=None, delete_after_upload=False):
        """Save an artifact (file/object) in TRAINS experiment storage.

        :param str name: Artifact name. Notice! it will override previous artifact if name already exists
        :param object artifact: Artifact object to upload. Currently supports:
            - string / pathlib2.Path are treated as path to artifact file to upload
                If wildcard or a folder is passed, zip file containing the local files will be created and uploaded
            - dict will be stored as .json file and uploaded
            - pandas.DataFrame will be stored as .csv.gz (compressed CSV file) and uploaded
            - numpy.ndarray will be stored as .npz and uploaded
            - PIL.Image will be stored to .png file and uploaded
        :param dict metadata: Simple key/value dictionary to store on the artifact
        :param bool delete_after_upload: If True local artifact will be deleted
            (only applies if artifact_object is a local file)
        """
        if not self._trains:
            return None
        self._trains.upload_artifact(name=name, artifact_object=artifact,
                                     metadata=metadata, delete_after_upload=delete_after_upload)

    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        if not self._trains:
            return None
        self._trains.close()
        self._trains = None

    @property
    def name(self):
        if not self._trains:
            return None
        return self._trains.name

    @property
    def version(self):
        if not self._trains:
            return None
        return self._trains.id

    def __getstate__(self):
        if not self._trains:
            return None
        return self._trains.id

    def __setstate__(self, state):
        self._rank = 0
        self._trains = None
        if state:
            self._trains = trains.Task.get_task(task_id=state)

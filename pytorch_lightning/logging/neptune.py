"""
Log using `neptune <https://www.neptune.ml>`_

Neptune logger can be used in the online mode or offline (silent) mode.
To log experiment data in online mode, NeptuneLogger requries an API key:

.. code-block:: python

    from pytorch_lightning.logging import NeptuneLogger
    # arguments made to NeptuneLogger are passed on to the neptune.experiments.Experiment class

    neptune_logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name="USER_NAME/PROJECT_NAME",
        experiment_name="default", # Optional,
        params={"max_epochs": 10}, # Optional,
        tags=["pytorch-lightning","mlp"] # Optional,
    )
    trainer = Trainer(max_epochs=10, logger=neptune_logger)

Use the logger anywhere in you LightningModule as follows:

.. code-block:: python

    def train_step(...):
        # example
        self.logger.experiment.log_metric("acc_train", acc_train) # log metrics
        self.logger.experiment.log_image("worse_predictions", prediction_image) # log images
        self.logger.experiment.log_artifact("model_checkpoint.pt", prediction_image) # log model checkpoint
        self.logger.experiment.whatever_neptune_supports(...)

    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.log_metric("acc_train", acc_train) # log metrics
        self.logger.experiment.log_image("worse_predictions", prediction_image) # log images
        self.logger.experiment.log_artifact("model_checkpoint.pt", prediction_image) # log model checkpoint
        self.logger.experiment.whatever_neptune_supports(...)


"""

from logging import getLogger

try:
    import neptune
except ImportError:
    raise ImportError('Missing neptune package. Run `pip install neptune-client`')

from torch import is_tensor

# from .base import LightningLoggerBase, rank_zero_only
from pytorch_lightning.logging.base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


class NeptuneLogger(LightningLoggerBase):
    def __init__(self, api_key=None, project_name=None, offline_mode=False,
                 experiment_name=None, upload_source_files=None,
                 params=None, properties=None, tags=None, **kwargs):
        """Initialize a neptune.ml logger.
        Requires either an API Key (online mode) or a local directory path (offline mode)

        :param str|None api_key: Required in online mode. Neputne API token, found on https://neptune.ml.
           Read how to get your API key https://docs.neptune.ml/python-api/tutorials/get-started.html#copy-api-token.
        :param str project_name: Required in online mode. Qualified name of a project in a form of
           "namespace/project_name" for example "tom/minst-classification".
           If None, the value of NEPTUNE_PROJECT environment variable will be taken.
           You need to create the project in https://neptune.ml first.
        :param bool offline_mode: Optional default False. If offline_mode=True no logs will be send to neptune.
           Usually used for debug purposes.
        :param str|None experiment_name: Optional. Editable name of the experiment.
           Name is displayed in the experiment’s Details (Metadata section) and in experiments view as a column.
        :param list|None upload_source_files: Optional. List of source files to be uploaded.
           Must be list of str or single str. Uploaded sources are displayed in the experiment’s Source code tab.
           If None is passed, Python file from which experiment was created will be uploaded.
           Pass empty list ([]) to upload no files. Unix style pathname pattern expansion is supported.
           For example, you can pass '*.py' to upload all python source files from the current directory.
           For recursion lookup use '**/*.py' (for Python 3.5 and later). For more information see glob library.
        :param dict|None params: Optional. Parameters of the experiment. After experiment creation params are read-only.
           Parameters are displayed in the experiment’s Parameters section and each key-value pair can be
           viewed in experiments view as a column.
        :param dict|None properties: Optional default is {}. Properties of the experiment.
           They are editable after experiment is created. Properties are displayed in the experiment’s Details and
           each key-value pair can be viewed in experiments view as a column.
        :param list|None tags: Optional default []. Must be list of str. Tags of the experiment.
           They are editable after experiment is created (see: append_tag() and remove_tag()).
           Tags are displayed in the experiment’s Details and can be viewed in experiments view as a column.
        """
        super().__init__()
        self.api_key = api_key
        self.project_name = project_name
        self.offline_mode = offline_mode
        self.experiment_name = experiment_name
        self.upload_source_files = upload_source_files
        self.params = params
        self.properties = properties
        self.tags = tags
        self._experiment = None
        self._kwargs = kwargs

        if offline_mode:
            self.mode = "offline"
            neptune.init(project_qualified_name='dry-run/project',
                         backend=neptune.OfflineBackend())
        else:
            self.mode = "online"
            neptune.init(api_token=self.api_key,
                         project_qualified_name=self.project_name)

        logger.info(f"NeptuneLogger was initialized in {self.mode} mode")

    @property
    def experiment(self):
        if self._experiment is not None:
            return self._experiment
        else:
            self._experiment = neptune.create_experiment(name=self.experiment_name,
                                                         params=self.params,
                                                         properties=self.properties,
                                                         tags=self.tags,
                                                         upload_source_files=self.upload_source_files,
                                                         **self._kwargs)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        for key, val in vars(params).items():
            self.experiment.set_property(f"param__{key}", val)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        """Log metrics (numeric values) in Neptune experiments

        :param float metric: Dictionary with metric names as keys and measured quanties as values
        :param int|None step: Step number at which the metrics should be recorded, must be strictly increasing

        """

        for key, val in metrics.items():
            if is_tensor(val):
                val = val.cpu().detach()

            if step is None:
                self.experiment.log_metric(key, val)
            else:
                self.experiment.log_metric(key, x=step, y=val)

    @rank_zero_only
    def finalize(self, status):
        self.experiment.stop()

    @property
    def name(self):
        if self.mode == "offline":
            return "offline-name"
        else:
            return self.experiment.name

    @property
    def version(self):
        if self.mode == "offline":
            return "offline-id-1234"
        else:
            return self.experiment.id

    @rank_zero_only
    def log_metric(self, metric_name, metric_value, step=None):
        """Log metrics (numeric values) in Neptune experiments

        :param str metric_name:  The name of log, i.e. mse, loss, accuracy.
        :param str metric_value: The value of the log (data-point).
        :param int|None step: Step number at which the metrics should be recorded, must be strictly increasing

        """
        if step is None:
            self.experiment.log_metric(metric_name, metric_value)
        else:
            self.experiment.log_metric(metric_name, x=step, y=metric_value)

    @rank_zero_only
    def log_text(self, log_name, text, step=None):
        """Log text data in Neptune experiment

        :param str log_name:  The name of log, i.e. mse, my_text_data, timing_info.
        :param str text: The value of the log (data-point).
        :param int|None step: Step number at which the metrics should be recorded, must be strictly increasing

        """
        if step is None:
            self.experiment.log_metric(log_name, text)
        else:
            self.experiment.log_metric(log_name, x=step, y=text)

    @rank_zero_only
    def log_image(self, log_name, image, step=None):
        """Log image data in Neptune experiment

        :param str log_name: The name of log, i.e. bboxes, visualisations, sample_images.
        :param str|PIL.Image|matplotlib.figure.Figure image: The value of the log (data-point).
           Can be one of the following types: PIL image, matplotlib.figure.Figure, path to image file (str)
        :param int|None step: Step number at which the metrics should be recorded, must be strictly increasing

        """
        if step is None:
            self.experiment.log_image(log_name, image)
        else:
            self.experiment.log_image(log_name, x=step, y=image)

    @rank_zero_only
    def log_artifact(self, artifact, destination=None):
        """Save an artifact (file) in Neptune experiment storage.

        :param str artifact: A path to the file in local filesystem.
        :param str|None destination: Optional default None.
           A destination path. If None is passed, an artifact file name will be used.

        """
        self.experiment.log_artifact(artifact, destination)

    @rank_zero_only
    def set_property(self, key, value):
        """Set key-value pair as Neptune experiment property.

        :param str key: Property key.
        :param obj value: New value of a property.

        """
        self.experiment.set_property(key, value)

    @rank_zero_only
    def append_tags(self, tags):
        """appends tags to neptune experiment

        :param str|tuple|list(str) tags: Tags to add to the current experiment.
           If str is passed, singe tag is added.
           If multiple - comma separated - str are passed, all of them are added as tags.
           If list of str is passed, all elements of the list are added as tags.

        """
        if not isinstance(tags, (list, set, tuple)):
            tags = [tags]  # make it as an iterable is if it is not yet
        self.experiment.append_tags(*tags)

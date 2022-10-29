import inspect
from typing import Any, Dict, TYPE_CHECKING, Union

from core.state import ProgressBarState, TrainerState

import pytorch_lightning as pl
from lightning_app.storage import Path
from lightning_app.utilities.app_helpers import Logger
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress.base import get_standard_metrics
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.parsing import collect_init_args

if TYPE_CHECKING:
    from core.components.script_runner import ScriptRunner


_log = Logger(__name__)


class PLAppProgressTracker(Callback):
    """This callback tracks and communicates the Trainer's progress to the running PyTorch Lightning App."""

    def __init__(self, work: "ScriptRunner", refresh_rate: int = 1) -> None:
        super().__init__()
        self.work = work
        self.refresh_rate = refresh_rate
        self.is_enabled = False
        self._state = ProgressBarState()

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str,
    ) -> None:
        self.is_enabled = trainer.is_global_zero

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # We calculate the estimated stepping batches here instead of in the setup hook, because calling the
        # `Trainer.estimated_stepping_batches` too early would lead to a barrier() call in case of DDP and since this
        # callback is only attached on rank 0, would lead to a stall.
        self._state.fit.estimated_stepping_batches = trainer.estimated_stepping_batches

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_: Any) -> None:
        self._state.fit.total_train_batches = self._total_train_batches(trainer)
        self._state.fit.total_val_batches = self._total_val_batches(trainer)
        self._state.fit.current_epoch = trainer.current_epoch
        if self.is_enabled:
            self._send_state()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_: Any) -> None:
        self._state.metrics = self._progress_bar_metrics(trainer, pl_module)
        current = self._train_batch_idx(trainer)
        self._state.fit.train_batch_idx = current
        self._state.fit.global_step = trainer.global_step
        if self._should_send(current, self._total_train_batches(trainer)):
            self._send_state()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.metrics = self._progress_bar_metrics(trainer, pl_module)
        if self.is_enabled:
            self._send_state()

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if trainer.state.fn == "fit":
            self._state.fit.val_dataloader_idx = dataloader_idx
            self._state.fit.total_val_batches = self._total_val_batches(trainer)
        if trainer.state.fn == "validate":
            self._state.val.dataloader_idx = dataloader_idx
            self._state.val.total_val_batches = self._total_val_batches(trainer)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_: Any) -> None:
        self._state.metrics = self._progress_bar_metrics(trainer, pl_module)
        current = self._val_batch_idx(trainer)
        if trainer.state.fn == "fit":
            self._state.fit.val_batch_idx = current
        if trainer.state.fn == "validate":
            self._state.val.val_batch_idx = current
        if self._should_send(current, self._total_val_batches(trainer)):
            self._send_state()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.metrics = self._progress_bar_metrics(trainer, pl_module)
        if self.is_enabled:
            self._send_state()

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._state.test.dataloader_idx = dataloader_idx
        self._state.test.total_test_batches = trainer.num_test_batches[dataloader_idx]

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._state.metrics = self._progress_bar_metrics(trainer, pl_module)
        current = self._test_batch_idx(trainer)
        self._state.test.test_batch_idx = current
        if self._should_send(current, trainer.num_test_batches[dataloader_idx]):
            self._send_state()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.metrics = self._progress_bar_metrics(trainer, pl_module)
        if self.is_enabled:
            self._send_state()

    def on_predict_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._state.predict.dataloader_idx = dataloader_idx
        self._state.predict.total_predict_batches = trainer.num_predict_batches[dataloader_idx]

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._state.metrics = self._progress_bar_metrics(trainer, pl_module)
        current = self._predict_batch_idx(trainer)
        self._state.predict.predict_batch_idx = current
        if self._should_send(current, trainer.num_predict_batches[dataloader_idx]):
            self._send_state()

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.metrics = self._progress_bar_metrics(trainer, pl_module)
        if self.is_enabled:
            self._send_state()

    def _train_batch_idx(self, trainer: "pl.Trainer") -> int:
        return trainer.fit_loop.epoch_loop.batch_progress.current.processed

    def _val_batch_idx(self, trainer: "pl.Trainer") -> int:
        if trainer.state.fn == "fit":
            loop = trainer.fit_loop.epoch_loop.val_loop
        else:
            loop = trainer.validate_loop

        current_batch_idx = loop.epoch_loop.batch_progress.current.processed
        return current_batch_idx

    def _test_batch_idx(self, trainer: "pl.Trainer") -> int:
        return trainer.test_loop.epoch_loop.batch_progress.current.processed

    def _predict_batch_idx(self, trainer: "pl.Trainer") -> int:
        return trainer.predict_loop.epoch_loop.batch_progress.current.processed

    def _total_train_batches(self, trainer: "pl.Trainer") -> Union[int, float]:
        return trainer.num_training_batches

    def _total_val_batches(self, trainer: "pl.Trainer") -> Union[int, float]:
        return sum(trainer.num_val_batches) if trainer.fit_loop.epoch_loop._should_check_val_epoch() else 0

    def _progress_bar_metrics(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> Dict[str, Union[str, float]]:
        standard_metrics = get_standard_metrics(trainer, pl_module)
        pbar_metrics = trainer.progress_bar_metrics
        return {**standard_metrics, **pbar_metrics}

    def _send_state(self) -> None:
        self.work.trainer_progress = self._state.dict()

    def _should_send(self, current: int, total: int) -> bool:
        return self.is_enabled and current % self.refresh_rate == 0 or current == total


class PLAppTrainerStateTracker(Callback):
    def __init__(self, work: "ScriptRunner") -> None:
        super().__init__()
        self.work = work
        self._state = TrainerState()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.fn = "fit"
        self.work.trainer_state = self._state.dict()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.fn = None
        self.work.trainer_state = self._state.dict()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.stage = "training"
        self.work.trainer_state = self._state.dict()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.stage = None
        self.work.trainer_state = self._state.dict()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.stage = "validating"
        self.work.trainer_state = self._state.dict()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.stage = None
        self.work.trainer_state = self._state.dict()

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.fn = "test"
        self._state.stage = "testing"
        self.work.trainer_state = self._state.dict()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.fn = None
        self._state.stage = None
        self.work.trainer_state = self._state.dict()

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.fn = "predict"
        self._state.stage = "predicting"
        self.work.trainer_state = self._state.dict()

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._state.fn = None
        self._state.stage = None
        self.work.trainer_state = self._state.dict()


class PLAppSummary(Callback):
    def __init__(self, work: "ScriptRunner") -> None:
        super().__init__()
        self.work = work

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        current_frame = inspect.currentframe()
        # Trainer.init() -> Trainer._call_callback_hooks() -> Callback.on_init_end()
        frame = current_frame.f_back.f_back
        init_args = {}
        for local_args in collect_init_args(frame, []):
            init_args.update(local_args)

        self.work.trainer_hparams = self._sanitize_trainer_init_args(init_args)

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str,
    ) -> None:
        self.work.model_hparams = self._sanitize_model_init_args(dict(**pl_module.hparams))

    def _sanitize_trainer_init_args(self, init_args: Dict[str, Any]) -> Dict[str, str]:
        if init_args["callbacks"]:
            init_args["callbacks"] = [c.__class__.__name__ for c in init_args["callbacks"]]
        init_args = {k: str(v) for k, v in init_args.items()}
        return init_args

    def _sanitize_model_init_args(self, init_args: Dict[str, Any]) -> Dict[str, str]:
        return {k: str(v) for k, v in init_args.items()}


class PLAppArtifactsTracker(Callback):
    def __init__(self, work: "ScriptRunner") -> None:
        super().__init__()
        self.work = work

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str,
    ) -> None:
        log_dir = self._get_logdir(trainer)
        self.work.log_dir = Path(log_dir) if log_dir is not None else None
        self._collect_logger_metadata(trainer)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.checkpoint_callback and trainer.checkpoint_callback.dirpath is not None:
            self.work.checkpoint_dir = Path(trainer.checkpoint_callback.dirpath)

    def _collect_logger_metadata(self, trainer: "pl.Trainer") -> None:
        if not trainer.loggers:
            return

        for logger in trainer.loggers:
            metadata = {"class_name": logger.__class__.__name__}
            if isinstance(logger, WandbLogger) and not logger._offline:
                metadata.update(
                    {"username": logger.experiment.entity, "project_name": logger.name, "run_id": logger.version}
                )

            if metadata and metadata not in self.work.logger_metadatas:
                self.work.logger_metadatas.append(metadata)

    @staticmethod
    def _get_logdir(trainer: "pl.Trainer") -> str:
        """The code here is the same as in the ``Trainer.log_dir``, with the exception of the broadcast call."""
        if len(trainer.loggers) == 1:
            if isinstance(trainer.logger, TensorBoardLogger):
                dirpath = trainer.logger.log_dir
            else:
                dirpath = trainer.logger.save_dir
        else:
            dirpath = trainer.default_root_dir
        return dirpath

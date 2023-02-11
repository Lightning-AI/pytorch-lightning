from typing import Any, List, Optional, Sequence, Union

from torch.utils.data import DataLoader

from lightning.pytorch.loops.dataloader.dataloader_loop import _DataLoaderLoop
from lightning.pytorch.loops.epoch.prediction_epoch_loop import _PredictionEpochLoop
from lightning.pytorch.loops.utilities import _no_grad_context, _set_sampler_epoch
from lightning.pytorch.strategies import DDPSpawnStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import _PREDICT_OUTPUT


class _PredictionLoop(_DataLoaderLoop):
    """Top-level loop where prediction starts.

    It simply iterates over each predict dataloader from one to the next by calling ``_PredictionEpochLoop.run()`` in
    its ``advance()`` method.
    """

    def __init__(self, inference_mode: bool = True) -> None:
        super().__init__()
        self.epoch_batch_indices: List[List[List[int]]] = []  # used by PredictionWriter
        self.epoch_loop = _PredictionEpochLoop()
        self.inference_mode = inference_mode

        self._results = None  # for `trainer._results` access
        self._predictions: List[List[Any]] = []  # num_dataloaders x batches
        self._return_predictions: bool = False

    @property
    def return_predictions(self) -> bool:
        """Whether to return the predictions or not."""
        return self._return_predictions

    @return_predictions.setter
    def return_predictions(self, return_predictions: Optional[bool] = None) -> None:
        # `DDPSpawnStrategy` plugins and derivatives don't support return predictions.
        is_ddp_spawn = isinstance(self.trainer.strategy, DDPSpawnStrategy)
        if return_predictions and is_ddp_spawn:
            raise MisconfigurationException(
                "`return_predictions` should be set to `False` when using the `DDPSpawnStrategy` or children class. "
                f"Found {return_predictions} with strategy {type(self.trainer.strategy)}."
            )
        # For non `DDPSpawnStrategy` plugin, the `return_predictions` is True by default unless user decide otherwise.
        self._return_predictions = not is_ddp_spawn if return_predictions is None else return_predictions
        self.epoch_loop.return_predictions = self._return_predictions

    @property
    def predictions(self) -> List[Any]:
        """The cached predictions."""
        if self._predictions == []:
            return self._predictions
        return self._predictions[0] if self.num_dataloaders == 1 else self._predictions

    @property
    def num_dataloaders(self) -> int:
        """Returns the number of prediction dataloaders."""
        # case where user does:
        # return dl1, dl2
        dataloaders = self.dataloaders
        length = len(dataloaders)
        if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
            length = len(dataloaders[0])
        return length

    @property
    def max_batches(self) -> List[Union[int, float]]:
        """The max number of batches this loop will run for each dataloader."""
        return self.trainer.num_predict_batches

    @property
    def dataloaders(self) -> Sequence[DataLoader]:
        """Returns all prediction dataloaders."""
        dataloaders = self.trainer.predict_dataloaders
        return [] if dataloaders is None else dataloaders

    @property
    def skip(self) -> bool:
        return sum(self.max_batches) == 0

    @_no_grad_context
    def run(self) -> Optional[_PREDICT_OUTPUT]:
        if self.skip:
            return None
        self.reset()
        self.on_run_start()
        while not self.done:
            try:
                self.on_advance_start()
                self.advance()
                self.on_advance_end()
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False
        return self.on_run_end()

    def reset(self) -> None:
        """Resets the internal state of the loop for a new run."""
        self._predictions = []
        self.epoch_batch_indices = []

        super().reset()
        # when restarting, if we are running twice, since there's no concept of `max_epochs` we need to reset the
        # current state when the loop has finished running
        if self.done:
            self.dataloader_progress.reset_on_run()

    def on_run_start(self) -> None:
        """Calls ``_on_predict_model_eval``, ``_on_predict_start`` and ``_on_predict_epoch_start`` hooks."""
        self.trainer._call_lightning_module_hook("on_predict_model_eval")
        self.trainer.lightning_module.zero_grad()
        self._on_predict_start()
        self._on_predict_epoch_start()

    def advance(self) -> None:
        """Predicts one entire dataloader."""
        dataloader = self.current_dataloader
        if dataloader is not None:
            _set_sampler_epoch(dataloader, self.trainer.fit_loop.epoch_progress.current.processed)
        dataloader = self.trainer.strategy.process_dataloader(dataloader)
        dataloader_iter = enumerate(dataloader)
        dl_max_batches = self.max_batches[self.current_dataloader_idx]

        dl_predictions, dl_batch_indices = self.epoch_loop.run(
            dataloader_iter, self.current_dataloader_idx, dl_max_batches, self.num_dataloaders
        )
        self._predictions.append(dl_predictions)
        self.epoch_batch_indices.append(dl_batch_indices)

    def on_run_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Calls ``on_predict_epoch_end`` and ``on_predict_end`` hooks and returns results from all dataloaders."""
        results = self._on_predict_epoch_end()
        self._on_predict_end()
        return results

    def teardown(self) -> None:
        pass

    def _on_predict_start(self) -> None:
        """Calls ``on_predict_start`` hooks."""
        self.trainer._call_callback_hooks("on_predict_start")
        self.trainer._call_lightning_module_hook("on_predict_start")
        self.trainer._call_strategy_hook("on_predict_start")

    def _on_predict_epoch_start(self) -> None:
        """Calls ``on_predict_epoch_start`` hooks."""
        self.trainer._call_callback_hooks("on_predict_epoch_start")
        self.trainer._call_lightning_module_hook("on_predict_epoch_start")

    def _on_predict_epoch_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Calls ``on_predict_epoch_end`` hook.

        Returns:
            the results for all dataloaders
        """
        self.trainer._call_callback_hooks("on_predict_epoch_end")
        self.trainer._call_lightning_module_hook("on_predict_epoch_end")

        if self.return_predictions:
            return self.predictions

    def _on_predict_end(self) -> None:
        """Resets previous gradient status and calls ``on_predict_end`` hook."""
        if not self.return_predictions:
            self._predictions = []
        self.epoch_batch_indices = []

        # hook
        self.trainer._call_callback_hooks("on_predict_end")
        self.trainer._call_lightning_module_hook("on_predict_end")
        self.trainer._call_strategy_hook("on_predict_end")

from typing import Any, List, Optional, Sequence

from deprecate.utils import void
from torch.utils.data import DataLoader

from pytorch_lightning.loops.dataloader.dataloader_loop import DataLoaderLoop
from pytorch_lightning.loops.epoch.prediction_epoch_loop import PredictionEpochLoop
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT


class PredictionLoop(DataLoaderLoop):
    """Loop to run over dataloaders for prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.predictions: List[List[Any]] = []
        self.epoch_batch_indices: List[List[int]] = []
        self.epoch_loop = PredictionEpochLoop()

        self._results = None  # for `trainer._results` access
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
                f"Found {return_predictions} with training_type_plugin {type(self.trainer.strategy)}."
            )
        # For non `DDPSpawnStrategy` plugin, the `return_predictions` is True by default unless user decide otherwise.
        self._return_predictions = not is_ddp_spawn if return_predictions is None else return_predictions

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
    def max_batches(self) -> List[int]:
        """The max number of batches this loop will run for each dataloader."""
        return self.trainer.num_predict_batches

    @property
    def dataloaders(self) -> Sequence[DataLoader]:
        """Returns all prediction dataloaders."""
        return self.trainer.predict_dataloaders

    @property
    def skip(self) -> bool:
        return sum(self.max_batches) == 0

    def connect(self, epoch_loop: PredictionEpochLoop) -> None:  # type: ignore[override]
        """Connect the prediction epoch loop with this loop."""
        self.epoch_loop = epoch_loop

    def reset(self) -> None:
        """Resets the internal state of the loop for a new run."""
        self.predictions = []
        self.epoch_batch_indices = []

        super().reset()
        # when restarting, if we are running twice, since there's no concept of `max_epochs` we need to reset the
        # current state when the loop has finished running
        if self.done:
            self.dataloader_progress.reset_on_run()

    def on_run_start(self) -> None:  # type: ignore[override]
        """Calls ``_on_predict_start`` hook."""
        self._on_predict_start()

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Predicts one entire dataloader."""
        void(*args, **kwargs)
        dataloader = self.current_dataloader
        if (
            dataloader is not None
            and getattr(dataloader, "sampler", None)
            and callable(getattr(dataloader.sampler, "set_epoch", None))
        ):
            # set seed for distributed sampler (enables shuffling for each epoch)
            dataloader.sampler.set_epoch(self.trainer.fit_loop.epoch_progress.current.processed)
        dataloader = self.trainer.strategy.process_dataloader(dataloader)
        dataloader_iter = enumerate(dataloader)
        dl_max_batches = self.max_batches[self.current_dataloader_idx]

        dl_predictions, dl_batch_indices = self.epoch_loop.run(
            dataloader_iter, self.current_dataloader_idx, dl_max_batches, self.num_dataloaders, self.return_predictions
        )
        self.predictions.append(dl_predictions)
        self.epoch_batch_indices.append(dl_batch_indices)

    def on_run_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Calls ``on_predict_epoch_end`` and ``on_predict_end`` hooks and returns results from all dataloaders."""
        results = self._on_predict_epoch_end()
        self._on_predict_end()
        return results

    def _on_predict_start(self) -> None:
        """Sets model to eval mode and disables gradients.

        Also calls ``on_predict_start`` and ``on_predict_epoch_start`` hooks.
        """
        # enable eval mode + no grads
        self._on_predict_model_eval()
        self.trainer.lightning_module.zero_grad()

        # hook
        self.trainer._call_callback_hooks("on_predict_start")
        self.trainer._call_lightning_module_hook("on_predict_start")
        self.trainer._call_strategy_hook("on_predict_start")

        self.trainer._call_callback_hooks("on_predict_epoch_start")
        self.trainer._call_lightning_module_hook("on_predict_epoch_start")

    def _on_predict_epoch_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Calls ``on_predict_epoch_end`` hook.

        Returns:
            the results for all dataloaders
        """
        results = self.predictions

        self.trainer._call_callback_hooks("on_predict_epoch_end", results)
        self.trainer._call_lightning_module_hook("on_predict_epoch_end", results)

        if self.return_predictions:
            return results[0] if self.num_dataloaders == 1 else results

    def _on_predict_end(self) -> None:
        """Resets previous gradient status and calls ``on_predict_end`` hook."""
        # clear memory. the predictions are extracted in `on_predict_epoch_end`.
        self.predictions = []
        self.epoch_batch_indices = []

        # hook
        self.trainer._call_callback_hooks("on_predict_end")
        self.trainer._call_lightning_module_hook("on_predict_end")
        self.trainer._call_strategy_hook("on_predict_end")

    def _on_predict_model_eval(self) -> None:
        """Calls ``on_predict_model_eval`` hook."""
        model_ref = self.trainer.lightning_module
        model_ref.on_predict_model_eval()

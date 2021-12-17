# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Sequence, Union

from deprecate.utils import void
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning.loops.dataloader import DataLoaderLoop
from pytorch_lightning.loops.epoch import EvaluationEpochLoop
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, ResultCollection
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


class EvaluationLoop(DataLoaderLoop):
    """Loops over all dataloaders for evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_loop = EvaluationEpochLoop()

        self._results = ResultCollection(training=False)
        self._outputs: List[EPOCH_OUTPUT] = []
        self._max_batches: List[int] = []
        self._has_run: bool = False

    @property
    def num_dataloaders(self) -> int:
        """Returns the total number of dataloaders."""
        # case where user does:
        # return dl1, dl2
        dataloaders = self.dataloaders
        if dataloaders is None:
            return 0
        length = len(dataloaders)
        if length > 0 and isinstance(dataloaders[0], (list, tuple)):
            length = len(dataloaders[0])
        return length

    @property
    def dataloaders(self) -> Sequence[DataLoader]:
        """Returns the validation or test dataloaders."""
        dataloaders = self.trainer.test_dataloaders if self.trainer.testing else self.trainer.val_dataloaders
        if dataloaders is None:
            raise RuntimeError("Dataloaders should be available.")
        return dataloaders

    def connect(self, epoch_loop: EvaluationEpochLoop) -> None:  # type: ignore[override]
        """Connect the evaluation epoch loop with this loop."""
        self.epoch_loop = epoch_loop

    @property
    def done(self) -> bool:
        """Returns whether all dataloaders are processed or evaluation should be skipped altogether."""
        return super().done or self.skip

    @property
    def skip(self) -> bool:
        """Returns whether the evaluation should be skipped."""
        max_batches = self._get_max_batches()
        return sum(max_batches) == 0

    def reset(self) -> None:
        """Resets the internal state of the loop."""
        self._max_batches = self._get_max_batches()
        # bookkeeping
        self._outputs = []

        if isinstance(self._max_batches, int):
            self._max_batches = [self._max_batches] * len(self.dataloaders)

        super().reset()

    def on_skip(self) -> List:
        return []

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs the ``_on_evaluation_model_eval``, ``_on_evaluation_start`` and ``_on_evaluation_epoch_start``
        hooks."""
        void(*args, **kwargs)

        # hook
        self._on_evaluation_model_eval()
        self.trainer.lightning_module.zero_grad()
        self._on_evaluation_start()
        self._on_evaluation_epoch_start()

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs evaluation on one single dataloader."""
        void(*args, **kwargs)

        dataloader_idx = self.current_dataloader_idx
        dataloader = self.trainer.training_type_plugin.process_dataloader(self.current_dataloader)
        self.data_fetcher = dataloader = self.trainer._data_connector.get_profiled_dataloader(
            dataloader, dataloader_idx=dataloader_idx
        )
        dl_max_batches = self._max_batches[dataloader_idx]

        dl_outputs = self.epoch_loop.run(
            dataloader, dataloader_idx if self.num_dataloaders > 1 else None, dl_max_batches
        )

        # store batch level output per dataloader
        self._outputs.append(dl_outputs)

        if not self.trainer.sanity_checking:
            # indicate the loop has run
            self._has_run = True

    def on_run_end(self) -> List[_OUT_DICT]:
        """Runs the ``_on_evaluation_epoch_end`` hook."""
        outputs, self._outputs = self._outputs, []  # free memory

        # lightning module method
        self._evaluation_epoch_end(outputs)

        # hook
        self._on_evaluation_epoch_end()

        # log epoch metrics
        eval_loop_results = self.trainer.logger_connector.update_eval_epoch_metrics()

        # hook
        self._on_evaluation_end()

        # enable train mode again
        self._on_evaluation_model_train()

        return eval_loop_results

    def teardown(self) -> None:
        self._results.cpu()
        self.epoch_loop.teardown()

    def _get_max_batches(self) -> List[int]:
        """Returns the max number of batches for each dataloader."""
        if self.trainer.testing:
            max_batches = self.trainer.num_test_batches
        else:
            if self.trainer.sanity_checking:
                self.trainer.num_sanity_val_batches = [
                    min(self.trainer.num_sanity_val_steps, val_batches) for val_batches in self.trainer.num_val_batches
                ]
                max_batches = self.trainer.num_sanity_val_batches
            else:
                max_batches = self.trainer.num_val_batches
        return max_batches

    def _reload_evaluation_dataloaders(self) -> None:
        """Reloads dataloaders if necessary."""
        if self.trainer.testing:
            self.trainer.reset_test_dataloader()
        elif self.trainer.val_dataloaders is None or self.trainer._should_reload_dl_epoch:
            self.trainer.reset_val_dataloader()

    def _on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_start`` hooks."""
        assert self._results is not None
        self._results.to(device=self.trainer.lightning_module.device)

        if self.trainer.testing:
            self.trainer._call_callback_hooks("on_test_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_test_start", *args, **kwargs)
            self.trainer._call_ttp_hook("on_test_start", *args, **kwargs)
        else:
            self.trainer._call_callback_hooks("on_validation_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_validation_start", *args, **kwargs)
            self.trainer._call_ttp_hook("on_validation_start", *args, **kwargs)

    def _on_evaluation_model_eval(self) -> None:
        """Sets model to eval mode."""
        if self.trainer.testing:
            self.trainer._call_lightning_module_hook("on_test_model_eval")
        else:
            self.trainer._call_lightning_module_hook("on_validation_model_eval")

    def _on_evaluation_model_train(self) -> None:
        """Sets model to train mode."""
        if self.trainer.testing:
            self.trainer._call_lightning_module_hook("on_test_model_train")
        else:
            self.trainer._call_lightning_module_hook("on_validation_model_train")

    def _on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_end`` hook."""
        if self.trainer.testing:
            self.trainer._call_callback_hooks("on_test_end", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_test_end", *args, **kwargs)
            self.trainer._call_ttp_hook("on_test_end", *args, **kwargs)
        else:
            self.trainer._call_callback_hooks("on_validation_end", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_validation_end", *args, **kwargs)
            self.trainer._call_ttp_hook("on_validation_end", *args, **kwargs)

        # reset the logger connector state
        self.trainer.logger_connector.reset_results()

    def _on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_epoch_start`` and ``on_{validation/test}_epoch_start`` hooks."""
        self.trainer.logger_connector.on_epoch_start()
        self.trainer._call_callback_hooks("on_epoch_start", *args, **kwargs)
        self.trainer._call_lightning_module_hook("on_epoch_start", *args, **kwargs)

        if self.trainer.testing:
            self.trainer._call_callback_hooks("on_test_epoch_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_test_epoch_start", *args, **kwargs)
        else:
            self.trainer._call_callback_hooks("on_validation_epoch_start", *args, **kwargs)
            self.trainer._call_lightning_module_hook("on_validation_epoch_start", *args, **kwargs)

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]) -> None:
        """Runs ``{validation/test}_epoch_end``"""
        # inform logger the batch loop has finished
        self.trainer.logger_connector.epoch_end_reached()

        # with a single dataloader don't pass a 2D list
        output_or_outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )

        # call the model epoch end
        if self.trainer.testing:
            self.trainer._call_lightning_module_hook("test_epoch_end", output_or_outputs)
        else:
            self.trainer._call_lightning_module_hook("validation_epoch_end", output_or_outputs)

    def _on_evaluation_epoch_end(self) -> None:
        """Runs ``on_{validation/test}_epoch_end`` hook."""
        hook_name = "on_test_epoch_end" if self.trainer.testing else "on_validation_epoch_end"
        self.trainer._call_callback_hooks(hook_name)
        self.trainer._call_lightning_module_hook(hook_name)

        self.trainer._call_callback_hooks("on_epoch_end")
        self.trainer._call_lightning_module_hook("on_epoch_end")
        self.trainer.logger_connector.on_epoch_end()

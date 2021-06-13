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
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache


class EvaluationLoop(object):

    def __init__(self, trainer: 'pl.Trainer'):
        self.trainer: 'pl.Trainer' = trainer
        self.outputs: EPOCH_OUTPUT = []
        self.predictions: Optional[PredictionCollection] = None
        self.max_batches: Optional[List[Union[int, float]]] = None
        self.warning_cache = WarningCache()
        self.num_dataloaders: Optional[int] = None
        self._val_results = ResultCollection(training=False)
        self._test_results = ResultCollection(training=False)

    @property
    def results(self) -> Optional[ResultCollection]:
        if self.trainer.validating or self.trainer.sanity_checking:
            return self._val_results
        elif self.trainer.testing:
            return self._test_results
        return None

    def on_trainer_init(self) -> None:
        self.trainer.num_sanity_val_batches = []
        self.trainer.num_test_batches = []
        self.trainer.num_val_batches = []
        self.trainer.test_dataloaders = None
        self.trainer.val_dataloaders = None

        # .validate() and .test() set this when they load a checkpoint
        self.trainer.validated_ckpt_path = None
        self.trainer.tested_ckpt_path = None

        # when true, print evaluation results in .validate() and .test()
        self.trainer.verbose_evaluate = True

    def get_evaluation_dataloaders(self) -> Tuple[Optional[List[DataLoader]], List[Union[int, float]]]:
        model = self.trainer.lightning_module

        # select dataloaders
        if self.trainer.testing:
            self.trainer.reset_test_dataloader(model)

            dataloaders = self.trainer.test_dataloaders
            max_batches = self.trainer.num_test_batches
        else:
            # val
            if self.trainer.val_dataloaders is None or self.trainer.reload_dataloaders_every_epoch:
                self.trainer.reset_val_dataloader(model)
            if self.trainer.sanity_checking:
                self.trainer.num_sanity_val_batches = [
                    min(self.trainer.num_sanity_val_steps, val_batches) for val_batches in self.trainer.num_val_batches
                ]
                max_batches = self.trainer.num_sanity_val_batches
            else:
                max_batches = self.trainer.num_val_batches
            dataloaders = self.trainer.val_dataloaders
        return dataloaders, max_batches

    def should_skip_evaluation(self, max_batches: List[Union[int, float]]) -> bool:
        return sum(max_batches) == 0

    def on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        self.should_track_batch_outputs_for_epoch_end: bool = self._should_track_batch_outputs_for_epoch_end()

        assert self.results is not None
        self.results.to(device=self.trainer.lightning_module.device)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_start', *args, **kwargs)

    def on_evaluation_model_eval(self) -> None:
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_eval()
        else:
            model_ref.on_validation_model_eval()

    def on_evaluation_model_train(self) -> None:
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_train()
        else:
            model_ref.on_validation_model_train()

    def on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        if self.trainer.testing:
            self.trainer.call_hook('on_test_end', *args, **kwargs)
            if self.trainer.logger is not None:
                self.trainer.logger.finalize("success")
        else:
            self.trainer.call_hook('on_validation_end', *args, **kwargs)

        if self.trainer.state.fn != TrainerFn.FITTING:
            # summarize profile results
            self.trainer.profiler.describe()

        # reset any `torchmetrics.Metric` and the logger connector state
        self.trainer.logger_connector.reset(metrics=True)

    def reload_evaluation_dataloaders(self) -> None:
        model = self.trainer.lightning_module
        if self.trainer.testing:
            self.trainer.reset_test_dataloader(model)
        else:
            self.trainer.reset_val_dataloader(model)

    def setup(self, max_batches: List[Union[int, float]], dataloaders: List[DataLoader]) -> None:
        # bookkeeping
        self.outputs = []
        self.predictions = PredictionCollection(self.trainer.global_rank, self.trainer.world_size)

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        self.max_batches = max_batches
        self.num_dataloaders = self._get_num_dataloaders(dataloaders)

    def on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.logger_connector.on_epoch_start()
        self.trainer.call_hook('on_epoch_start', *args, **kwargs)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_epoch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_start', *args, **kwargs)

    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Dict[str, Union[Any, int]]:
        # make dataloader_idx arg in validation_step optional
        step_kwargs = OrderedDict([('batch', batch), ('batch_idx', batch_idx)])

        multiple_val_loaders = (
            not self.trainer.testing and self._get_num_dataloaders(self.trainer.val_dataloaders) > 1
        )
        multiple_test_loaders = (self.trainer.testing and self._get_num_dataloaders(self.trainer.test_dataloaders) > 1)

        if multiple_test_loaders or multiple_val_loaders:
            step_kwargs['dataloader_idx'] = dataloader_idx

        return step_kwargs

    def _get_num_dataloaders(self, dataloaders: Optional[List[DataLoader]]) -> int:
        # case where user does:
        # return dl1, dl2
        if dataloaders is not None:
            length = len(dataloaders)
            if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
                length = len(dataloaders[0])
            return length
        else:
            return 0

    def evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Optional[STEP_OUTPUT]:
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        if self.trainer.testing:
            self.trainer.lightning_module._current_fx_name = "test_step"
            with self.trainer.profiler.profile("test_step"):
                output = self.trainer.accelerator.test_step(step_kwargs)
        else:
            self.trainer.lightning_module._current_fx_name = "validation_step"
            with self.trainer.profiler.profile("validation_step"):
                output = self.trainer.accelerator.validation_step(step_kwargs)

        return output

    def evaluation_step_end(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        if self.trainer.testing:
            output = self.trainer.call_hook('test_step_end', *args, **kwargs)
        else:
            output = self.trainer.call_hook('validation_step_end', *args, **kwargs)
        return output

    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        model = self.trainer.lightning_module
        if self.trainer.testing:
            return is_overridden('test_epoch_end', model)
        else:
            return is_overridden('validation_epoch_end', model)

    def evaluation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # inform logger the batch loop has finished
        self.trainer.logger_connector.epoch_end_reached()

        # call the model epoch end
        model = self.trainer.lightning_module

        # unset dataloader_idx in model
        model._current_dataloader_idx = None

        if self.trainer.testing:
            if is_overridden('test_epoch_end', model):
                model._current_fx_name = 'test_epoch_end'
                model.test_epoch_end(outputs)

        else:
            if is_overridden('validation_epoch_end', model):
                model._current_fx_name = 'validation_epoch_end'
                model.validation_epoch_end(outputs)

    def on_evaluation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.trainer.logger_connector.on_batch_start()

        # set dataloader_idx to model and track batch_size
        assert self.num_dataloaders is not None
        self.trainer.logger_connector.on_evaluation_batch_start(batch, batch_idx, dataloader_idx, self.num_dataloaders)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_batch_start', batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook('on_validation_batch_start', batch, batch_idx, dataloader_idx)

    def on_evaluation_batch_end(
        self,
        output: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.trainer.testing:
            self.trainer.call_hook('on_test_batch_end', output, batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook('on_validation_batch_end', output, batch, batch_idx, dataloader_idx)

        self.trainer.logger_connector.on_batch_end()

        # store predicitons if do_write_predictions and track eval loss history
        self.store_predictions(output, batch_idx, dataloader_idx)

    def store_predictions(self, output: Optional[STEP_OUTPUT], batch_idx: int, dataloader_idx: int) -> None:
        # Add step predictions to prediction collection to write later
        if output is not None and self.predictions is not None:
            if isinstance(output, ResultCollection) and self.trainer.testing:
                self.predictions.add(output.pop('predictions', None))

        # track debug metrics
        self.trainer.dev_debugger.track_eval_loss_history(batch_idx, dataloader_idx, output)

    def on_evaluation_epoch_end(self) -> None:
        hook_name = "on_test_epoch_end" if self.trainer.testing else "on_validation_epoch_end"
        self.trainer.call_hook(hook_name)
        self.trainer.call_hook('on_epoch_end')
        self.trainer.logger_connector.on_epoch_end()

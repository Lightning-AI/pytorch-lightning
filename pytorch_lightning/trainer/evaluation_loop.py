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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from pytorch_lightning.core.step_result import Result
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.warnings import WarningCache

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from pytorch_lightning import Trainer
    from pytorch_lightning.accelerators.accelerator import _STEP_OUTPUT_TYPE


class EvaluationLoop(object):

    def __init__(self, trainer: 'Trainer'):
        self.trainer: 'Trainer' = trainer
        self.outputs: List['_STEP_OUTPUT_TYPE'] = []
        self.predictions: Optional[PredictionCollection] = None
        self.max_batches: Optional[List[int, float]] = None
        self.warning_cache: WarningCache = WarningCache()
        self.num_dataloaders: int = None

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

    def get_evaluation_dataloaders(self) -> Tuple[List['DataLoader'], List[Union[int, float]]]:
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
        if self.trainer.testing:
            self.trainer.call_hook('on_test_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_start', *args, **kwargs)

    def on_evaluation_model_eval(self, *_: Any, **__: Any) -> None:
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_eval()
        else:
            model_ref.on_validation_model_eval()

    def on_evaluation_model_train(self, *_: Any, **__: Any) -> None:
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_train()
        else:
            model_ref.on_validation_model_train()

    def on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        if self.trainer.testing:
            self.trainer.call_hook('on_test_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_end', *args, **kwargs)

        if self.trainer.state != TrainerState.FITTING:
            # summarize profile results
            self.trainer.profiler.describe()

    def reload_evaluation_dataloaders(self) -> None:
        model = self.trainer.lightning_module
        if self.trainer.testing:
            self.trainer.reset_test_dataloader(model)
        else:
            self.trainer.reset_val_dataloader(model)

    def setup(self, max_batches: List[Union[int, float]], dataloaders: List['DataLoader']) -> None:
        # bookkeeping
        self.outputs = []
        self.predictions = PredictionCollection(self.trainer.global_rank, self.trainer.world_size)

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        self.max_batches = max_batches
        self.num_dataloaders = self._get_num_dataloaders(dataloaders)
        self._predictions = [[] for _ in range(self.num_dataloaders)]

    def on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.call_hook('on_epoch_start', *args, **kwargs)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_epoch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_start', *args, **kwargs)

    def _build_args(self, batch: Any, batch_idx: int, dataloader_idx: int) -> List[Union[Any, int]]:
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        multiple_val_loaders = (
            not self.trainer.testing and self._get_num_dataloaders(self.trainer.val_dataloaders) > 1
        )
        multiple_test_loaders = (self.trainer.testing and self._get_num_dataloaders(self.trainer.test_dataloaders) > 1)

        if multiple_test_loaders or multiple_val_loaders:
            args.append(dataloader_idx)

        return args

    def _get_num_dataloaders(self, dataloaders: List['DataLoader']) -> int:
        # case where user does:
        # return dl1, dl2
        length = len(dataloaders)
        if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
            length = len(dataloaders[0])
        return length

    def evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> '_STEP_OUTPUT_TYPE':
        # configure args
        args = self._build_args(batch, batch_idx, dataloader_idx)

        model_ref = self.trainer.lightning_module
        model_ref._results = Result()

        if self.trainer.testing:
            model_ref._current_fx_name = "test_step"
            with self.trainer.profiler.profile("test_step"):
                output = self.trainer.accelerator.test_step(args)
        else:
            model_ref._current_fx_name = "validation_step"
            with self.trainer.profiler.profile("validation_step"):
                output = self.trainer.accelerator.validation_step(args)

        # capture any logged information
        self.trainer.logger_connector.cache_logged_metrics()
        # track batch size for weighted average
        is_result_obj = isinstance(output, Result)
        if is_result_obj:
            output.track_batch_size(batch)

        return output

    def evaluation_step_end(self, *args: Any, **kwargs: Any) -> '_STEP_OUTPUT_TYPE':
        if self.trainer.testing:
            output = self.trainer.call_hook('test_step_end', *args, **kwargs)
        else:
            output = self.trainer.call_hook('validation_step_end', *args, **kwargs)
        return output

    def evaluation_epoch_end(self, outputs: List['_STEP_OUTPUT_TYPE']) -> None:
        # unset dataloder_idx in model
        self.trainer.logger_connector.evaluation_epoch_end()

        # call the model epoch end
        model = self.trainer.lightning_module

        if self.trainer.testing:
            if is_overridden('test_epoch_end', model=model):
                model._current_fx_name = 'test_epoch_end'
                model.test_epoch_end(outputs)

        else:
            if is_overridden('validation_epoch_end', model=model):
                model._current_fx_name = 'validation_epoch_end'
                model.validation_epoch_end(outputs)

        # capture logging
        self.trainer.logger_connector.cache_logged_metrics()

    def on_evaluation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # set dataloader_idx to model and track batch_size
        self.trainer.logger_connector.on_evaluation_batch_start(batch, dataloader_idx, self.num_dataloaders)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_batch_start', batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook('on_validation_batch_start', batch, batch_idx, dataloader_idx)

    def on_evaluation_batch_end(
        self,
        output: '_STEP_OUTPUT_TYPE',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.trainer.testing:
            self.trainer.call_hook('on_test_batch_end', output, batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook('on_validation_batch_end', output, batch, batch_idx, dataloader_idx)

        # store predicitons if do_write_predictions and track eval loss history
        self.store_predictions(output, batch_idx, dataloader_idx)

    def store_predictions(self, output: '_STEP_OUTPUT_TYPE', batch_idx: int, dataloader_idx: int) -> None:
        # Add step predictions to prediction collection to write later
        if output is not None:
            do_write_predictions = isinstance(output, Result) and self.trainer.testing
            if do_write_predictions:
                self.predictions.add(output.pop('predictions', None))

        # track debug metrics
        self.trainer.dev_debugger.track_eval_loss_history(batch_idx, dataloader_idx, output)

    def on_evaluation_epoch_end(self, outputs: Union[List[List[Dict]], List[Dict]]) -> None:
        model_ref = self.trainer.lightning_module
        hook_name = "on_test_epoch_end" if self.trainer.testing else "on_validation_epoch_end"

        self.trainer._reset_result_and_set_hook_fx_name(hook_name)

        with self.trainer.profiler.profile(hook_name):

            if hasattr(self.trainer, hook_name):
                on_evaluation_epoch_end_hook = getattr(self.trainer, hook_name)
                on_evaluation_epoch_end_hook(outputs)

            if is_overridden(hook_name, model_ref):
                model_hook_fx = getattr(model_ref, hook_name)
                if is_param_in_hook_signature(model_hook_fx, "outputs"):
                    model_hook_fx(outputs)
                else:
                    self.warning_cache.warn(
                        f"`ModelHooks.{hook_name}` signature has changed in v1.3. `outputs` parameter has been added."
                        " Support for the old signature will be removed in v1.5", DeprecationWarning
                    )
                    model_hook_fx()

        self.trainer._cache_logged_metrics()

        self.trainer.call_hook('on_epoch_end')

    def log_evaluation_step_metrics(self, output: '_STEP_OUTPUT_TYPE', batch_idx: int) -> None:
        if self.trainer.sanity_checking:
            return

        step_log_metrics = {}
        step_pbar_metrics = {}

        self.__log_result_step_metrics(step_log_metrics, step_pbar_metrics, batch_idx)

    def __log_result_step_metrics(
        self, step_log_metrics: Dict[str, Any], step_pbar_metrics: Dict[str, Any], batch_idx: int
    ) -> None:
        cached_results = self.trainer.logger_connector.cached_results
        cached_batch_pbar_metrics, cached_batch_log_metrics = cached_results.update_logger_connector()

        step_log_metrics.update(cached_batch_log_metrics)
        step_pbar_metrics.update(cached_batch_pbar_metrics)

        if len(step_log_metrics) > 0:
            # make the metrics appear as a different line in the same graph
            metrics_by_epoch = {}
            for k, v in step_log_metrics.items():
                metrics_by_epoch[f'{k}/epoch_{self.trainer.current_epoch}'] = v

            self.trainer.logger_connector.log_metrics(metrics_by_epoch, {}, step=batch_idx)

        if len(step_pbar_metrics) > 0:
            self.trainer.logger_connector.add_progress_bar_metrics(step_pbar_metrics)
